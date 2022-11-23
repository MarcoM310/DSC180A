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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from models import VGG
from train import train1Epoch
from train import test1Epoch
from src.helper import datasetCreator

torch.cuda.empty_cache()
import seaborn as sns

test_path = "/home/jmryan/private/DSC180/A/test/testdata.csv"
train_path = "/home/jmryan/private/DSC180/A/train/traindata.csv"
val_path = "/home/jmryan/private/DSC180/A/val/valdata.csv"


def run_all(df_train, df_val):
    train_dataset = datasetCreator(df=df_train)
    val_dataset = datasetCreator(df=df_val)
    train_dl = DataLoader(train_dataset, batch_size=16, num_workers=0, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=16, num_workers=0, shuffle=False)

    model = resnet152(weights=ResNet152_Weights.DEFAULT)

    if torch.cuda.is_available():
        dev = "gpu"
    else:
        dev = "cpu"

    net.train()
    trainer.fit(net, train_dl, val_dl)

    plt.plot(np.arange(len(net.val_loss_epoch) - 1), net.val_loss_epoch[1:])
    plt.plot(np.arange(len(net.train_loss_epoch)), net.train_loss_epoch)
    plt.legend(["Val", "Train"])
    plt.show()


def main(targets):
    if targets[0] == "test":
        df_train, df_val = train_test_split(
            pd.read_csv(test_path, index_col=0), test_size=0.2
        )
    elif targets[0] == "train":
        df_train = pd.read_csv(train_path, index_col=0)
        df_val = pd.read_csv(val_path, index_col=0)

    run_all(df_train, df_val, BATCH_SIZE)


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
