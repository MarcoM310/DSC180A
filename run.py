import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import datetime
from tqdm import tqdm
import sys

import torch
from torch import nn
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet152
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

import os
import cv2
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import pearsonr

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

# from torch.utils.tensorboard import SummaryWriter
# import torch.optim as optim

from src.models.train_model import train1Epoch, trainAndSave
from src.models.predict_model import test1Epoch
from src.data import make_dataset
from src.features.build_features import files2df, PreprocessedImageDataset, Loader
from src.helper.models import VGG
from src.visualization.visualize import PlotTrainValLoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
test_path = "/home/ddavilag/private/DSC180A_Final/DSC180A/test/testdata.csv"
train_path = "/home/ddavilag/private/DSC180A_Final/DSC180A/data/out/traindata.csv"
val_path = "/home/ddavilag/private/DSC180A_Final/DSC180A/data/out/valdata.csv"
LR = 0.0001
EPOCHS = 5


def run_all(df_val, df_train=None):
    ### 1) if train != None: use train set (on train mode)
    # 2) save model
    # 3) use model on val set (on eval mode)
    # 4) output predictions/visualizations
    # Otherwise, use pretrained model on the test set (on eval mode, ~100 rows), output predictions/visualizations

    valid_set = PreprocessedImageDataset(df=df_val)
    valid_loader = Loader(valid_set, mode="eval")

    if df_train is not None:
        # training brand new ResNet model

        train_set = PreprocessedImageDataset(df=df_train)
        train_loader = Loader(train_set, mode="train")
        # print(next(iter(train_loader)))
        resnet = resnet152(pretrained=True)
        resnet.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
        resnet.to(DEVICE)
        # model = VGG("VGG16").to(DEVICE)
        trainAndSave(resnet, train_loader, valid_loader)
        print("trained and saved model!")
    else:

        # TODO: use pretrained model
        resnet = resnet152(pretrained=True)  # weights="ResNet152_Weights.DEFAULT")
        resnet.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
        resnet.to(DEVICE)
        optimizer = optim.Adam(resnet.parameters(), lr=LR)

        checkpoint = torch.load("src/models/resnet152.pt")
        resnet.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        loss_fn = nn.L1Loss().to(DEVICE)
        print("loaded pretrained model")

        resnet.eval()
        for param in resnet.parameters():
            param.requires_grad = False
        with torch.no_grad():
            test_loss = test1Epoch(0, resnet, loss_fn, valid_loader)
            print(f"Overall Test Loss: {test_loss}")


def main(targets):
    if targets[0] == "test":
        print("Testing pretrained model on test set...")
        ### if test, use pretrained model on the test set (on eval mode, ~100 rows), output predictions/visualizations
        df_test = pd.read_csv(test_path, index_col=0)
        df_test = df_test.head(5)
        print(df_test.head())
        run_all(df_test)
    elif targets[0] == "train":
        print("Training model on train set...")
        ### 1) if train, use train set (on train mode), save model
        # 2) use model on val set (on eval mode), output predictions/visualizations
        df_train = pd.read_csv(train_path, index_col=0)
        df_val = pd.read_csv(val_path, index_col=0)
        df_train = df_train.head(5)
        df_val = df_val.head(5)
        print(df_train.head(), df_val.head())
        run_all(df_train, df_val)


if __name__ == "__main__":
    ### Run with `python run.py test` or `python run.py train`
    ### target: train or test
    print("Running run.py...")
    print(DEVICE)
    targets = sys.argv[1:]
    main(targets)
