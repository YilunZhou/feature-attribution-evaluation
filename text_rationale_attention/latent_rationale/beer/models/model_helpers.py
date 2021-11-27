#!/usr/bin/env python

from latent_rationale.beer.models.simpleclassifier import SimpleClassifier
from latent_rationale.beer.models.latent import LatentRationaleModel
from latent_rationale.beer.models.rl import RLModel


def build_model(model_type, vocab, cfg=None):

    emb_size = cfg["emb_size"]  # default = 200
    hidden_size = cfg["hidden_size"]  # default = 200
    dropout = cfg["dropout"]  # default = 0.2
    layer = cfg["layer"]  # default = "rcnn"
    vocab_size = len(vocab.w2i)
    dependent_z = cfg["dependent_z"]  # default = True

    if model_type == "baseline":
        return SimpleClassifier(
            vocab_size, emb_size=emb_size, hidden_size=hidden_size, vocab=vocab, dropout=dropout, layer=layer)
    elif model_type == "rl":
        selection = cfg["selection"]  # default = 0.13
        reg_strength = cfg["reg_strength"]  # default = 1.0
        return RLModel(
            vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, vocab=vocab, dropout=dropout,
            dependent_z=dependent_z, layer=layer, selection=selection, reg_strength=reg_strength)
    elif model_type == "latent":
        selection = cfg["selection"]  # default = 0.13
        reg_strength = cfg["reg_strength"]  # default = 1.0
        return LatentRationaleModel(
            vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size, vocab=vocab, dropout=dropout,
            dependent_z=dependent_z, layer=layer, selection=selection, reg_strength=reg_strength)
    else:
        raise ValueError("Unknown model")
