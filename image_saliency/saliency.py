import random, os
from tqdm import tqdm, trange

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_image import LimeImageExplainer
import shap

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder


def saliency_grad(net, img, pil_transform=None, label=None, return_label=False, abs_val=True, sum_channel=True):
    '''
    img can be either a PIL Image or a torch Tensor
    if img is a PIL Image, then pil_transform is applied to turn the image into a torch Tensor
    '''
    assert isinstance(img, Image.Image) or isinstance(img, torch.Tensor)
    if isinstance(img, Image.Image):
        img = pil_transform(img).unsqueeze(0)
    device = next(net.parameters()).device
    img = img.to(device)
    img.requires_grad = True
    logits = net(img)
    if label is None:
        label = logits.detach().cpu().numpy().flatten().argmax()
    logit = logits[0, label]
    logit.backward()
    grad = img.grad[0].detach().cpu().numpy()
    if abs_val:
        grad = abs(grad)
    if sum_channel:
        grad = grad.sum(axis=0)
    if not return_label:
        return grad
    else:
        return grad, label


def saliency_grad_input(net, img, pil_transform=None, label=None, return_label=False, abs_val=True, sum_channel=True):
    '''
    img can be either a PIL Image or a torch Tensor
    if img is a PIL Image, then pil_transform is applied to turn the image into a torch Tensor
    '''
    grad, label = saliency_grad(net, img, pil_transform=pil_transform, label=label, return_label=True, abs_val=False,
                                sum_channel=False)
    if isinstance(img, Image.Image):
        img = pil_transform(img).numpy()
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    grad_input = grad * img
    if abs_val:
        grad_input = abs(grad_input)
    if sum_channel:
        grad_input = grad_input.sum(axis=0)
    if not return_label:
        return grad_input
    else:
        return grad_input, label


def saliency_smooth_grad(net, img, pil_transform=None, label=None, N_repeat=50, noise_level=0.15, return_label=False,
                         abs_val=True, sum_channel=True):
    assert isinstance(img, Image.Image) or isinstance(img, torch.Tensor)
    if isinstance(img, Image.Image):
        img = pil_transform(img).unsqueeze(0)
    _, _, H, W = img.shape
    device = next(net.parameters()).device
    if label is None:
        with torch.no_grad():
            img = img.to(device)
            logits = net(img)
            label = logits.detach().cpu().numpy().flatten().argmax()
    sigma = noise_level * (img.max() - img.min())
    noises = torch.randn(N_repeat, 3, H, W).to(device) * sigma
    imgs = (torch.cat([img] * N_repeat, dim=0) + noises).to(device)
    imgs.requires_grad = True
    logits = net(imgs)
    logit = logits[:, label].sum()
    logit.backward()
    grads = imgs.grad.detach().cpu().numpy()
    if abs_val:
        grads = abs(grads)
    grad = grads.mean(axis=0)
    if sum_channel:
        grad = grad.sum(axis=0)
    if not return_label:
        return grad
    else:
        return grad, label


def saliency_smooth_grad_input(net, img, pil_transform=None, label=None, return_label=False, abs_val=True,
                               sum_channel=True):
    '''
    img can be either a PIL Image or a torch Tensor
    if img is a PIL Image, then pil_transform is applied to turn the image into a torch Tensor
    '''
    grad, label = saliency_smooth_grad(net, img, pil_transform=pil_transform, label=label, return_label=True,
                                       abs_val=False, sum_channel=False)
    if isinstance(img, Image.Image):
        img = pil_transform(img).numpy()
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    grad_input = grad * img
    if abs_val:
        grad_input = abs(grad_input)
    if sum_channel:
        grad_input = grad_input.sum(axis=0)
    if not return_label:
        return grad_input
    else:
        return grad_input, label


def saliency_gradcam(net, img, pil_transform=None, label=None, return_label=False, **kwargs):
    '''
    img can be either a PIL Image or a torch Tensor
    if img is a PIL Image, then pil_transform is applied to turn the image into a torch Tensor
    note that the code below technically implements CAM, but GradCAM reduces to CAM for fully convolutional networks like resnet
    '''
    assert isinstance(img, Image.Image) or isinstance(img, torch.Tensor)
    if isinstance(img, Image.Image):
        img = pil_transform(img).unsqueeze(0)
    _, _, H, W = img.shape
    device = next(net.parameters()).device
    img = img.to(device)
    with torch.no_grad():
        if label is None:
            label = net(img).cpu().numpy().flatten().argmax()
        conv_maps = net.maxpool(net.relu(net.bn1(net.conv1(img))))
        conv_maps = net.layer4(net.layer3(net.layer2(net.layer1(conv_maps)))).cpu().numpy()[0]
        weights = net.fc.weight.data.cpu().numpy()[label].reshape(-1, 1, 1)
    avg_conv_map = (conv_maps * weights).sum(axis=0)
    avg_conv_map[avg_conv_map < 0] = 0
    avg_conv_map = np.array(Image.fromarray(avg_conv_map).resize((H, W)))
    if not return_label:
        return avg_conv_map
    else:
        return avg_conv_map, label


def saliency_lime(net, img, pil_transform=None, label=None, return_label=False, **kwargs):
    assert isinstance(img, Image.Image), 'LIME input has to be a PIL Image'
    if label is None:
        with torch.no_grad():
            device = next(net.parameters()).device
            img_torch = pil_transform(img).unsqueeze(0).to(device)
            label = net(img_torch).cpu().numpy().flatten().argmax()
    def predict_proba(imgs):
        with torch.no_grad():
            device = next(net.parameters()).device
            imgs = [pil_transform(im).to(device) for im in imgs]
            imgs = torch.stack(imgs, dim=0)
            prob = F.softmax(net(imgs), dim=-1).cpu().numpy()
        return prob
    explainer = LimeImageExplainer()
    img = np.array(img)
    explanation = explainer.explain_instance(img, predict_proba, labels=(label, ),
                    hide_color=0, top_labels=None, num_samples=200, batch_size=200, progress_bar=False)
    coefs = {k: v for k, v in explanation.local_exp[label]}
    attr_map = np.zeros_like(explanation.segments).astype('float')
    for i in range(explanation.segments.max() + 1):
        attr_map[explanation.segments==i] = abs(coefs[i])
    if return_label:
        return attr_map, label
    else:
        return attr_map


def saliency_shap(net, img, pil_transform=None, label=None, return_label=False, abs_val=True, sum_channel=True, **kwargs):
    '''kwargs is required to have a key of 'background_imgs' that is an N x 3 x 224 x 224 tensor of test set images'''
    device = next(net.parameters()).device
    if isinstance(img, Image.Image):
        img = pil_transform(img).unsqueeze(0)
    if label is None:
        with torch.no_grad():
            img = img.to(device)
            logits = net(img)
            label = logits.detach().cpu().numpy().flatten().argmax()
    class MyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
        def forward(self, x):
            logits = self.net(x)[:, label].unsqueeze(1)
            return logits
    explainer = shap.GradientExplainer(MyNet(), kwargs['background_imgs'].to(device), batch_size=128, num_outputs=1)
    shap_img = img.to(device)
    shap_v = explainer.shap_values(shap_img, rseed=0)
    saliency = shap_v.squeeze()
    if abs_val:
        saliency = abs(saliency)
    if sum_channel:
        saliency = saliency.sum(axis=0)
    if return_label:
        return saliency, label
    else:
        return saliency
