
import os, argparse
from collections import defaultdict
from tqdm import tqdm, trange
import numpy as np

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim
from torch.optim import Adam
from torch.utils.data import DataLoader

from latent_rationale.beer.constants import PAD_TOKEN
from latent_rationale.beer.vocabulary import Vocabulary
from latent_rationale.beer.util import print_parameters, load_embeddings

from data_utils import load_data


def get_args():
    parser = argparse.ArgumentParser(description="Beer multi-aspect sentiment analysis")

    # Beer specific arguments
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--embeddings', type=str, default="embedding.txt.gz")
    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--save_path', type=str, default='results')
    # general arguments
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--emb_size', type=int, default=200)
    parser.add_argument('--hidden_size', type=int, default=200)
    # optimization
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    # model / layer
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    args = parser.parse_args()
    return args


def masked_softmax(attn_odds, lens):
    for i in range(attn_odds.shape[0]):
        attn_odds[i, lens[i]:] = -float('inf')
    attn = nn.Softmax(dim=-1)(attn_odds)
    return attn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn1 = nn.Linear(hidden_size, hidden_size // 2)
        self.attn2 = nn.Linear(hidden_size // 2, 1, bias=False)

    def forward(self, hidden, lens):
        # hidden: B x L x H, lens: list of length B
        attn1 = nn.Tanh()(self.attn1(hidden))  # B x L x (H / 2)
        attn2 = self.attn2(attn1).squeeze(-1)  # B x L
        attn = masked_softmax(attn2, lens)
        return attn

class AttentionLSTMClassifier(nn.Module):
    def __init__(self, hidden_size, emb_size, vocab, num_classes=2):
        super().__init__()
        vocab_size = len(vocab.w2i)
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=1)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.output = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, tokenss, return_attn=False):
        assert isinstance(tokenss, list)
        lens = [len(tks) for tks in tokenss]
        padded_tokenss = pad_sequence(tokenss, padding_value=1, batch_first=True)
        padded_embss = self.emb(padded_tokenss)
        packed_embss = pack_padded_sequence(padded_embss, lens, batch_first=True, enforce_sorted=False)
        packed_output, (_, _) = self.lstm(packed_embss)
        padded_output, lens2 = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0)
        assert list(lens2.cpu().numpy()) == lens, 'lens mismatch? '
        attn_weights = self.attention(padded_output, lens)
        context_vec = (attn_weights.unsqueeze(-1) * padded_output).sum(dim=1)
        if not return_attn:
            return self.output(context_vec)
        else:
            return self.output(context_vec), attn_weights

def custom_collate_fn(batch):
    tokenss, ys = zip(*batch)
    return tokenss, ys

def evaluate(model, vocab, data_split, batch_size=256, device=None):
    """
    Loss of a model on given data set (using minibatches)
    Also computes some statistics over z assignments.
    """
    model.eval()  # disable dropout
    total = defaultdict(float)
    total_examples = 0
    total_correct = 0
    total_loss = 0
    data_loader = DataLoader(data_split, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(data_loader, ncols=70):
            wordss, labels = batch
            tokenss = [torch.tensor([vocab.w2i.get(w, 0) for w in words]).to(device) for words in wordss]
            model.zero_grad()
            preds = model(tokenss)
            loss_val = loss(preds, torch.tensor(labels).to(device))
            total_examples += len(labels)
            total_loss += loss_val.item() * len(labels)
            total_correct += (preds.cpu().numpy().argmax(axis=1) == labels).sum()
    return total_correct / total_examples, total_loss / total_examples

def train():
    cfg = get_args()
    cfg = vars(cfg)
    device = torch.device(cfg['device'])
    print("device:", device)
    data_name = cfg['data_fn'][:-4]
    save_path = os.path.join(cfg["save_path"], data_name, 'attention')
    os.makedirs(save_path, exist_ok=True)
    log_file = open(os.path.join(save_path, 'progress.log'), 'w')
    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, str(v)))
    num_epochs = cfg['num_epochs']
    batch_size = cfg["batch_size"]
    data = load_data(cfg['data_fn'])
    print("Loading pre-trained word embeddings")
    vocab = Vocabulary()
    vectors = load_embeddings(cfg["embeddings"], vocab)
    model = AttentionLSTMClassifier(cfg['hidden_size'], cfg['emb_size'], vocab)
    with torch.no_grad():
        model.emb.weight.data.copy_(torch.from_numpy(vectors))
        print("Embeddings fixed: {}".format(cfg["fix_emb"]))
        model.emb.weight.requires_grad = not cfg["fix_emb"]
    model = model.to(device)
    optimizer = Adam(model.parameters(), weight_decay=cfg["weight_decay"])
    loss = nn.CrossEntropyLoss()
    # print model and parameters
    print(model)
    print_parameters(model)

    best_acc = None
    pad_idx = vocab.w2i[PAD_TOKEN]

    # main training loop
    for epoch in range(num_epochs):
        model.train()
        train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        for batch in tqdm(train_loader, ncols=70):
            # forward pass
            wordss, labels = batch
            tokenss = [torch.tensor([vocab.w2i.get(w, 0) for w in words]).to(device) for words in wordss]
            model.zero_grad()
            preds = model(tokenss)
            loss_val = loss(preds, torch.tensor(labels).to(device))
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["max_grad_norm"])
            optimizer.step()

        # evaluate
        model.eval()
        dev_acc, dev_loss = evaluate(model, vocab, data['val'], batch_size=batch_size, device=device)
        log_str = f'dev acc: {dev_acc:0.3f}, loss: {dev_loss:0.3f}'
        print(log_str)
        log_file.write(log_str + '\n')
        log_file.flush()
        if best_acc is None or dev_acc > best_acc:
            best_acc = dev_acc
            ckpt = {
                "state_dict": model.state_dict(), "cfg": cfg,
                "optimizer_state_dict": optimizer.state_dict()
            }
            path = os.path.join(save_path, 'best.pt')
            torch.save(ckpt, path)
        if dev_acc >= 0.97:
            break

if __name__ == "__main__":
    train()
