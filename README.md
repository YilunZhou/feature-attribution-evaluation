# An Evaluation for Feature Attribution Methods

This is the code repository for the paper [_Do Feature Attribution Methods Correctly Attribute Features?_](https://arxiv.org/abs/2104.14403) by Yilun Zhou, Serena Booth, Marco Tulio Ribeiro and Julie Shah.

## Requirements
This repository has minimal dependencies. Specifically, it requires `tqdm` for the progress bar, `numpy` and `torch` for numerical computing, `matplotlib` for plotting, `Pillow` for image manipulation, and `lime` and `shap` for the implementation of LIME and SHAP methods. All dependencies can be installed with `pip install -r requirements.txt`.

## Code Structure
This repository contains two folders.
* `image_saliency` contains code related to training and explaining the ResNet-34 model on our bird dataset.
* `text_rationale_attention` contains code related to training and explaining rationale and attention models on the modified BeerAdvocate dataset.

Please refer to the `README.md` in the respective folder for detailed information.

## Logistics

For any questions, please contact Yilun Zhou at [yilun@mit.edu](mailto:yilun@mit.edu). The paper can be cited as

```
@article{zhou2021feature,
  title={Do Feature Attribution Methods Correctly Attribute Features?},
  author={Zhou, Yilun and Booth, Serena and Ribeiro, Marco Tulio and Shah, Julie},
  journal={arXiv preprint arXiv:2104.14403},
  year={2021}
}
```
