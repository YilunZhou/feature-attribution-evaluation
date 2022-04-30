# An Evaluation Method for Feature Attribution Explainers

This is the code repository for the AAAI 2022 paper [_Do Feature Attribution Methods Correctly Attribute Features?_](https://arxiv.org/abs/2104.14403) by Yilun Zhou, Serena Booth, Marco Tulio Ribeiro and Julie Shah. A video presentation is available [here](https://www.youtube.com/watch?v=kAodFw6jvvo).

**TLDR:** We "unit test" several popular feature attribution algorithms for CV and NLP models to see if they can identify features known to be highly important to model predictions; (un)surprisingly, they mostly can't.

**What's Next:** Don't stop at local explanations. Ensuring that high-quality model understanding is derived from these local explanations is equally important. Check out our [ExSum framework](https://yilunzhou.github.io/exsum/) for more details on this aspect.

## Requirements
This repository has minimal dependencies. Specifically, it requires `tqdm` for the progress bar, `numpy` and `torch` for numerical computing, `matplotlib` for plotting, `Pillow` for image manipulation, and `lime` and `shap` for the implementation of LIME and SHAP methods. All dependencies can be installed with `pip install -r requirements.txt`.

## Code Structure
This repository contains two folders.
* `image_saliency` contains code related to training and explaining the ResNet-34 model on our bird dataset.
* `text_rationale_attention` contains code related to training and explaining rationale and attention models on the modified BeerAdvocate dataset.

Please refer to the `README.md` in the respective folder for detailed information.

## Contact and Citation

For any questions, please contact Yilun Zhou at [yilun@mit.edu](mailto:yilun@mit.edu). The paper can be cited as

```
@inproceedings{zhou2022feature,
    title = {Do Feature Attribution Methods Correctly Attribute Features?},
    author = {Zhou, Yilun and Booth, Serena and Ribeiro, Marco Tulio and Shah, Julie},
    booktitle = {Proceedings of the 36th AAAI Conference on Artificial Intelligence},
    year = {2022},
    month = {Feb},
    publisher = {AAAI}
}
```
