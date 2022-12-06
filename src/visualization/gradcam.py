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
from torchvision.models import resnet152
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import argparse

from models import VGG, ResNet
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="gradcam", choices=["gradcam"])
    parser.add_argument(
        "--image-path", type=str, default="./examples/both.png", help="Input image path"
    )
    parser.add_argument(
        "--aug_smooth",
        action="store_true",
        help="Apply test time augmentation to smooth the CAM",
    )
    parser.add_argument(
        "--eigen_smooth",
        action="store_true",
        help="Reduce noise by taking the first principle componenet"
        "of cam_weights*activations",
    )

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


args = get_args()
rgb_img = np.float32((valid_set[0][0]).T)
