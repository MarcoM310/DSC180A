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
from datasetCreator import ImageSubset

torch.cuda.empty_cache()
import seaborn as sns

test_df = pd.read_csv(
    "/home/ddavilag/teams/dsc-180a---a14-[88137]/BNPP_DT_test_with_ages.csv",
    usecols=cols,
).set_index("unique_key")
train_df = pd.read_csv(
    "/home/ddavilag/teams/dsc-180a---a14-[88137]/BNPP_DT_train_with_ages.csv",
    usecols=cols,
).set_index("unique_key")
val_df = pd.read_csv(
    "/home/ddavilag/teams/dsc-180a---a14-[88137]/BNPP_DT_val_with_ages.csv",
    usecols=cols,
).set_index("unique_key")


# test_df = pd.read_csv('/home/mmorocho/teams/dsc-180a---a14-[88137]/BNPP_DT_test_with_ages.csv', usecols = cols).set_index('unique_key')
# train_df = pd.read_csv('/home/mmorocho/teams/dsc-180a---a14-[88137]/BNPP_DT_train_with_ages.csv', usecols = cols).set_index('unique_key')
# val_df = pd.read_csv('/home/mmorocho/teams/dsc-180a---a14-[88137]/BNPP_DT_val_with_ages.csv', usecols = cols).set_index('unique_key')


def run_all(df_train, df_val, BATCH_SIZE):
    print(BATCH_SIZE)
    train_dataset = PreprocessedImageDataset(df=df_train.to_numpy())
    val_dataset = PreprocessedImageDataset(df=df_val.to_numpy())
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


def main():
    args = sys.argv[1:]
    if args[0] == "test":
        df_train, df_val = train_test_split(
            pd.read_csv(TEST_PATH, index_col=0), test_size=0.2
        )
    elif args[0] == "train":
        df_train = pd.read_csv(TRAIN_PATH, index_col=0)
        df_val = pd.read_csv(VAL_PATH, index_col=0)

    BATCH_SIZE = eval(args[1])
    run_all(df_train, df_val, BATCH_SIZE)


if __name__ == "__main__":
    main()
