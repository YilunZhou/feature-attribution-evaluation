#!/usr/bin/env python

import torch
from torch import nn
from latent_rationale.common.util import get_z_stats
from latent_rationale.common.classifier import Classifier
from latent_rationale.common.latent import DependentLatentModel
from torch.nn.functional import softplus, sigmoid, tanh

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Model device:", device)


__all__ = ["LatentRationaleModel"]


class LatentRationaleModel(nn.Module):
    """
    Latent Rationale

    Consists of:

    p(y | x, z)     observation model / classifier
    p(z | x)        latent model

    In case of full VI:
    q(z | x, y)     inference model

    """
    def __init__(self,
                 vocab:          object = None,
                 vocab_size:     int = 0,
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 dropout:        float = 0.1,
                 layer:          str = "rcnn",
                 dependent_z:    bool = True,
                 z_rnn_size:     int = 30,
                 selection:      float = 0.13,
                 reg_strength:   float = 1.0,
                 ):

        super(LatentRationaleModel, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab = vocab

        self.selection = selection
        self.reg_strength = reg_strength

        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, dropout=dropout, layer=layer)

        if self.dependent_z:
            self.latent_model = DependentLatentModel(
                embed=embed, hidden_size=hidden_size,
                dropout=dropout, layer=layer)
        else:
            raise ValueError("Independent not implemented")

        self.criterion = nn.CrossEntropyLoss()

    @property
    def z(self):
        return self.latent_model.z

    @property
    def z_layer(self):
        return self.latent_model.z_layer

    @property
    def z_dists(self):
        return self.latent_model.z_dists

    def predict(self, x, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"

        with torch.no_grad():
            mask = (x != 1)
            predictions = self.forward(x)
            num_0, num_c, num_1, total = get_z_stats(self.z, mask)
            selected = num_1 / float(total)
            optional = dict(selected=selected)
            return predictions, optional

    def forward(self, x):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)  # [B,T]
        z = self.latent_model(x, mask)
        # print(set(list(z.detach().cpu().numpy().flat)))
        y = self.classifier(x, mask, z)

        return y

    def get_loss(self, preds, targets, mask=None):

        optional = {}
        nll = self.criterion(preds, targets)
        optional["nll"] = nll.item()

        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # L0 regularizer (sparsity constraint)
        # pre-compute for regularizers: pdf(0.)
        z_dists = self.latent_model.z_dists
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)
        pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

        pdf_nonzero = 1. - pdf0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `self.selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        c0_hat = self.reg_strength * abs(l0 - self.selection)
        loss = nll + c0_hat
        optional["c0_hat"] = c0_hat.item()

        # z statistics
        num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, mask)
        optional["p0"] = num_0 / float(total)
        optional["pc"] = num_c / float(total)
        optional["p1"] = num_1 / float(total)
        optional["selected"] = optional["pc"] + optional["p1"]

        return loss, optional
