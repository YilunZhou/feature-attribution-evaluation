
import os, time, datetime, json
from collections import defaultdict

from tqdm import tqdm, trange

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, \
    ExponentialLR
from torch.utils.data import DataLoader

from latent_rationale.beer.constants import PAD_TOKEN
from latent_rationale.beer.models.model_helpers import build_model
from latent_rationale.beer.vocabulary import Vocabulary
from latent_rationale.beer.util import \
    get_args, prepare_minibatch, \
    print_parameters, load_embeddings, \
    initialize_model_
from latent_rationale.common.util import make_kv_string

from data_utils import load_data

def custom_collate_fn(batch):
    tokenss, ys = zip(*batch)
    return tokenss, ys

def evaluate_loss(model, data_split, batch_size=256, device=None):
    """
    Loss of a model on given data set (using minibatches)
    Also computes some statistics over z assignments.
    """
    model.eval()  # disable dropout
    total = defaultdict(float)
    total_examples = 0
    total_correct = 0
    data_loader = DataLoader(data_split, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    for batch in tqdm(data_loader, ncols=70):
        x, targets, _ = prepare_minibatch(batch, model.vocab, device=device)
        mask = (x != 1)
        batch_examples = len(targets)
        total_examples += batch_examples
        with torch.no_grad():
            output = model(x)
            preds = output.cpu().numpy().argmax(axis=1)
            total_correct += (preds == targets.cpu().numpy()).sum()
            loss, loss_opt = model.get_loss(output, targets, mask=mask)
            total["loss"] += loss.item() * batch_examples
            # e.g. mse_loss, loss_z_x, sparsity_loss, coherence_loss
            for k, v in loss_opt.items():
                total[k] += v * batch_examples
    result = {}
    for k, v in total.items():
        if not k.startswith("z_num"):
            result[k] = v / float(total_examples)
    if "z_num_1" in total:
        z_total = total["z_num_0"] + total["z_num_c"] + total["z_num_1"]
        selected = total["z_num_1"] / float(z_total)
        result["p1r"] = selected
        result["z_num_0"] = total["z_num_0"]
        result["z_num_c"] = total["z_num_c"]
        result["z_num_1"] = total["z_num_1"]
    assert 'acc' not in result, 'acc already a key?'
    result['acc'] = total_correct / total_examples
    return result

def train():
    """
    Main training loop.
    """

    cfg = get_args()
    cfg = vars(cfg)

    device = torch.device(cfg['device'])
    print("device:", device)

    data_name = cfg['data_fn'][:-4]
    save_path = os.path.join(cfg["save_path"], data_name, f'sel_{cfg["selection"]:0.2f}', cfg["model"])
    os.makedirs(save_path, exist_ok=True)
    log_file = open(os.path.join(save_path, 'progress.log'), 'w')

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, str(v)))

    num_epochs = cfg['num_epochs']  # default = 100
    batch_size = cfg["batch_size"]  # default = 256

    data = load_data(cfg['data_fn'])

    print("Loading pre-trained word embeddings")
    vocab = Vocabulary()
    vectors = load_embeddings(cfg["embeddings"], vocab)  # embeddings default = "data/beer/review+wiki.filtered.200.txt.gz"

    # build model
    model = build_model(cfg["model"], vocab, cfg=cfg)  # model is "rl" or "latent"
    initialize_model_(model)

    # load pre-trained word embeddings
    with torch.no_grad():
        model.embed.weight.data.copy_(torch.from_numpy(vectors))
        print("Embeddings fixed: {}".format(cfg["fix_emb"]))
        model.embed.weight.requires_grad = not cfg["fix_emb"]  # fix_emb default to True unless --train_embed is enabled

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"],  # lr default = 0.0004
                     weight_decay=cfg["weight_decay"])  # weight_decay default = 2e-6

    # print model and parameters
    print(model)
    print_parameters(model)

    start = time.time()
    best_loss = None
    best_nll = None
    best_acc = None
    pad_idx = vocab.w2i[PAD_TOKEN]

    # main training loop
    for epoch in range(num_epochs):
        model.train()
        train_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        for batch in tqdm(train_loader, ncols=70):
            # forward pass
            x, targets, _ = prepare_minibatch(batch, model.vocab, device=device)
            model.zero_grad()
            output = model(x)
            mask = (x != pad_idx)
            assert pad_idx == 1, "pad idx"
            loss, loss_optional = model.get_loss(output, targets, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg["max_grad_norm"])  # max_grad_norm default = 5.0
            optimizer.step()

        # evaluate
        model.eval()
        dev_eval = evaluate_loss(model, data['val'], batch_size=batch_size, device=device)
        cur_acc, cur_nll, cur_loss = dev_eval['acc'], dev_eval['nll'], dev_eval['loss']
        sel_str = f', selected: {dev_eval["selected"]:0.3f}' if "selected" in dev_eval else ''
        log_str = f'dev acc: {cur_acc:0.3f}, nll: {cur_nll:0.3f}, loss: {cur_loss:0.3f}{sel_str}'
        print(log_str)
        log_file.write(log_str + '\n')
        log_file.flush()
        # save best model parameters
        if best_loss is None or cur_loss < best_loss:
            best_loss = cur_loss
            ckpt = {
                "state_dict": model.state_dict(), "cfg": cfg,
                "best_loss": best_loss, "best_loss_epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict()
            }
            path = os.path.join(save_path, 'best_loss.pt')
            torch.save(ckpt, path)
        if best_nll is None or cur_nll < best_nll:
            best_nll = cur_nll
            ckpt = {
                "state_dict": model.state_dict(), "cfg": cfg,
                "best_nll": best_nll, "best_nll_epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict()
            }
            path = os.path.join(save_path, 'best_nll.pt')
            torch.save(ckpt, path)
        if best_acc is None or cur_acc > best_acc:
            best_acc = cur_acc
            ckpt = {
                "state_dict": model.state_dict(), "cfg": cfg,
                "best_acc": best_acc, "best_acc_epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict()
            }
            path = os.path.join(save_path, 'best_acc.pt')
            torch.save(ckpt, path)
        if dev_eval['acc'] >= 0.97 and abs(dev_eval["selected"] - cfg['selection']) <= 0.01:
            ckpt = {
                "state_dict": model.state_dict(), "cfg": cfg,
                "optimizer_state_dict": optimizer.state_dict()
            }
            path = os.path.join(save_path, f'done.pt')
            torch.save(ckpt, path)
            break
        else:
            ckpt = {
            "state_dict": model.state_dict(), "cfg": cfg,
            "optimizer_state_dict": optimizer.state_dict()
            }
            path = os.path.join(save_path, f'epoch_{epoch}.pt')
            torch.save(ckpt, path)

if __name__ == "__main__":
    train()
