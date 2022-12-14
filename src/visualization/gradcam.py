import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import datetime
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
from torchvision.models import model152
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse

from models import VGG, resnet
from train import train1Epoch, test1Epoch

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
import seaborn as sns

import os
import cv2
import argparse

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    RawScoresOutputTarget,
)
from pytorch_grad_cam.utils.image import show_cam_on_image


def gradcam_viz(resnet, test_set, severity_level):
    """
    Visualize GradCAM for a given resnet and test set.
    'severity_level' is one of [normal, mild, moderate, severe].
    """
    if severity_level == "normal":
        input_tensor = test_set[7][0].unsqueeze(0).to(DEVICE)
        rgb_img = np.float32((test_set[7][0]).T)
    elif severity_level == "mild":
        input_tensor = test_set[1][0].unsqueeze(0).to(DEVICE)
        rgb_img = np.float32((test_set[1][0]).T)
    elif severity_level == "moderate":
        input_tensor = test_set[0][0].unsqueeze(0).to(DEVICE)
        rgb_img = np.float32((test_set[0][0]).T)
    elif severity_level == "severe":
        input_tensor = test_set[23][0].unsqueeze(0).to(DEVICE)
        rgb_img = np.float32((test_set[23][0]).T)
    else:
        raise ValueError(
            "severity_level must be one of [normal, mild, moderate, severe]."
        )

    for param in resnet.parameters():
        param.requires_grad = True
    resnet.eval()
    target_layers = [resnet.layer4[-1]]
    targets = [RawScoresOutputTarget()]

    target_layers = [resnet.layer4]
    with GradCAM(model=resnet, target_layers=target_layers) as cam:
        grayscale_cams = cam(
            input_tensor=input_tensor, targets=targets, aug_smooth=True
        )
        cam_image = show_cam_on_image(rgb_img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255 * grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255 * rgb_img), cam, cam_image))
    print(f"{severity_level} Case GradCAM")
    output = (
        Image.fromarray(images)
        .transpose(Image.ROTATE_270)
        .transpose(Image.FLIP_LEFT_RIGHT)
    )
    output = output.save(f"images/gradCAM_{severity_level}.png")
