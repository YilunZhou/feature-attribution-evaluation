# Rationale and Attention Experiments

This repository contains code to train the rationale and attention models, as introduced below.  

## Rationale Model
To train a rationale model, run
```bash
python rationale_model.py --data_fn DATA_FN --model MODEL --selection SEL --device cuda
```
where `DATA_FN` is one of the three pickled dataset files in the `datasets` folder (e.g. `datasets/article_dataset.pkl`), `MODEL` is either `rl` (for RL model) or `latent` (for CR model), and `SEL` is the target selection rate as a decimal number (e.g. `0.07`). If GPU is not available, please change `cuda` to `cpu`. The result will be saved to a folder named `results/DATA_FN/sel_SEL/MODEL/`. The validation set statistics will be written to `progess.log` under that folder.

## Attention Model
To train an attention model, run
```bash
python attention_model.py --data_fn DATA_FN --device cuda
```
where `DATA_FN` is one of the three dataset files. If GPU is not available, please change `cuda` to `cpu`. The result will be saved to a folder named `results/DATA_FN/attention`. The validation set statistics will be written to `progess.log` under that folder.

## Acknowledgement
The rationale training code is based on the [repository](https://github.com/bastings/interpretable_predictions) accompanying the paper [_Interpretable Neural Predictions with Differentiable Binary Variables_](https://www.aclweb.org/anthology/P19-1284) by Jasmijn Bastings, Wilker Aziz, and Ivan Titov. The regularization is modified to use a target selection rate and disable the discontinuity penalty.
