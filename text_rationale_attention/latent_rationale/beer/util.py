import os
import argparse
import random
import gzip
import json
import numpy as np
import torch
from collections import namedtuple
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn
from latent_rationale.beer.constants import UNK_TOKEN, PAD_TOKEN
import math


def decorate_token(t, z_):
    dec = "**" if z_ == 1 else "__" if z_ > 0 else ""
    return dec + t + dec


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def find_ckpt_in_directory(path):
    for f in os.listdir(os.path.join(path, "")):
        if f.startswith('model'):
            return os.path.join(path, f)
    print("Could not find ckpt in {}".format(path))


def filereader(path):
    """read SST lines"""
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


BeerExample = namedtuple("Example", ["tokens", "label"])
BeerTestExample = namedtuple("Example", ["tokens", "label", "annotation"])


def beer_reader(path, pos_scores, neg_scores, aspect=0, max_len=0):
    """
    Reads in Beer multi-aspect sentiment data
    :param path:
    :param aspect: which aspect to train/evaluate (-1 for all)
    :return:
    """
    num_pos = 0
    num_neg = 0
    data = []
    assert aspect == 0, 'aspect must be 0'
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            scores = list(map(float, parts[:5]))
            score = scores[aspect]
            if score in pos_scores:
                num_pos += 1
                label = 1
            elif score in neg_scores:
                num_neg += 1
                label = 0
            else:
                continue
            tokens = parts[5:]
            if max_len > 0:
                tokens = tokens[:max_len]
            data.append(BeerExample(tokens=tokens, label=label))
    print('num_pos:', num_pos, 'num_neg:', num_neg)
    return data


def beer_annotations_reader(path, pos_scores, neg_scores, aspect=0):
    """
    Reads in Beer annotations from json
    :param path:
    :param aspect: which aspect to evaluate
    :return:
    """
    num_pos = 0
    num_neg = 0
    examples = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = data["x"]
            scores = data["y"]
            annotations = [data["0"], data["1"], data["2"],
                           data["3"], data["4"]]
            score = scores[aspect]
            if score in pos_scores:
                num_pos += 1
                label = 1
            elif score in neg_scores:
                num_neg += 1
                label = 0
            else:
                continue
            annotation = annotations[aspect]
            examples.append(BeerTestExample(tokens=tokens, label=label, annotation=annotation))
    print('num_pos:', num_pos, 'num_neg:', num_neg)
    return examples


def print_parameters(model):
    """Prints model parameters"""
    total = 0
    total_wo_embed = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        total_wo_embed += np.prod(p.shape) if "embed" not in name else 0
        print("{:30s} {:14s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}".format(total))
    print("Total parameters (w/o embed): {}\n".format(total_wo_embed))


def load_embeddings(path, vocab, dim=200):
    """
    Load word embeddings and update vocab.
    :param path: path to word embedding file
    :param vocab:
    :param dim: dimensionality of the pre-trained embeddings
    :return:
    """
    if not os.path.exists(path):
        raise RuntimeError("You need to download the word embeddings. "
                           "See `data/beer` for the download script.")
    vectors = []
    w2i = {}
    i2w = []

    # Random embedding vector for unknown words
    vectors.append(np.random.uniform(
        -0.05, 0.05, dim).astype(np.float32))
    w2i[UNK_TOKEN] = 0
    i2w.append(UNK_TOKEN)

    # Zero vector for padding
    vectors.append(np.zeros(dim).astype(np.float32))
    w2i[PAD_TOKEN] = 1
    i2w.append(PAD_TOKEN)

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            w2i[word] = len(vectors)
            i2w.append(word)
            v = np.array(vec.split(), dtype=np.float32)
            assert len(v) == dim, "dim mismatch"
            vectors.append(v)

    vocab.w2i = w2i
    vocab.i2w = i2w

    return np.stack(vectors)


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    tokenss, ys = mb
    lengths = np.array([len(tokens) for tokens in tokenss])
    maxlen = lengths.max()
    reverse_map = None

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in tokens], maxlen) for tokens in tokenss]

    x = np.array(x)
    y = np.array(ys, dtype='int64')

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # create reverse map
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y, reverse_map


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model_(model):
    """
    Model initialization.

    :param model:
    :return:
    """
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))


def get_predict_args():
    parser = argparse.ArgumentParser(
        description="Beer multi-aspect sentiment prediction")
    parser.add_argument('--ckpt', type=str, default="path_to_checkpoint",
                        required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--plot', action="store_true", default=False)
    args = parser.parse_args()
    return args


def get_args():
    parser = argparse.ArgumentParser(
        description="Beer multi-aspect sentiment analysis")

    # Beer specific arguments
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--embeddings', type=str,
                        default="embedding.txt.gz",
                        help="path to external embeddings")
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--max_len', type=int, default=256,
                        help="maximum input length (cut off afterwards)")
    parser.add_argument('--save_path', type=str, default='results')

    # general arguments
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--emb_size', type=int, default=200)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--attention_size', type=int, default=100)

    # optimization
    parser.add_argument('--weight_decay', type=float, default=2e-6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=0.97)
    parser.add_argument('--scheduler',
                        choices=["plateau", "multistep", "exponential"],
                        default="exponential")
    parser.add_argument('--milestones', nargs='+', type=int,
                        default=[25, 50, 75],
                        help="epochs after which to reduce learning rate "
                             "when using multistep scheduler")
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help="significant change threshold for lr scheduler")
    parser.add_argument('--patience', type=int, default=5,
                        help="patience for lr scheduler")
    parser.add_argument('--cooldown', type=int, default=0,
                        help="cooldown for lr scheduler")

    # model / layer
    parser.add_argument('--layer', choices=["lstm", "rcnn"], default="rcnn")
    parser.add_argument('--model', choices=["baseline", "latent", "rl"],
                        default="baseline")
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--dist', choices=["", "hardkuma"], default="")

    parser.add_argument('--no-dependent-z', action='store_false', dest='dependent_z', 
                        help="make dependent decisions about z using an LSTM")

    # regularization for latent model
    parser.add_argument('--selection', type=float, default=0.13,
                        help="select this ratio of input words "
                             "(e.g. 0.13 for 13%)")
    parser.add_argument('--reg_strength', type=float, default=1.0,
                        help="Regularization strength")

    args = parser.parse_args()
    return args
