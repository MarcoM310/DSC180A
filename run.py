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
import torchvision
import torchvision.transforms as transforms

# from torch.utils.tensorboard import SummaryWriter
# import torch.optim as optim

from src.models.train_model import train1Epoch, trainAndSave
from src.models.predict_model import test1Epoch
from src.data import make_dataset
from src.features.build_features import files2df, PreprocessedImageDataset, Loader
from src.helper.models import VGG
from src.visualization.visualize import PlotTrainValLoss

torch.cuda.empty_cache()
import seaborn as sns

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
        resnet = resnet152(pretrained=True)
        resnet.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
        resnet.to(DEVICE)
        # model = VGG("VGG16").to(DEVICE)
        trainAndSave(resnet, train_loader, valid_loader)
    else:

        # TODO: use pretrained model
        resnet = torch.load("resnet152.pt")
        resnet.eval()


def main(targets):
    if targets[0] == "test":
        ### if test, use pretrained model on the test set (on eval mode, ~100 rows), output predictions/visualizations
        df_test = pd.read_csv(test_path, index_col=0)
        run_all(df_test)
    elif targets[0] == "train":
        ### 1) if train, use train set (on train mode), save model
        # 2) use model on val set (on eval mode), output predictions/visualizations
        df_train = pd.read_csv(train_path, index_col=0)
        df_val = pd.read_csv(val_path, index_col=0)
        run_all(df_train, df_val)


if __name__ == "__main__":
    ### Run with `python run.py test` or `python run.py train`
    ### target: train or test
    targets = sys.argv[1:]
    main(targets)
