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

DATA_PATH = "/home/ddavilag/teams/dsc-180a---a14-[88137]/df_bnpp_datapaths.csv"
KEY_PATH = "/home/ddavilag/teams/dsc-180a---a14-[88137]/df_bnpp_keys.csv"

df_datapaths = pd.read_csv(DATA_PATH, header=None).T.merge(
    pd.read_csv(KEY_PATH, header=None).T, left_index=True, right_index=True
)
df_datapaths.columns = ["filepaths", "key"]
df_datapaths.key = df_datapaths.key.apply(lambda x: eval(x))
df_datapaths.filepaths = df_datapaths.filepaths.apply(lambda x: eval(x))
df_datapaths = df_datapaths.set_index("key")
# missing h5py files 7-9

cols = ["unique_key", "bnpp_value_log", "BNP_value"]
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
train_df["heart"] = train_df["BNP_value"].apply(lambda x: int(x > 400))
test_df["heart"] = test_df["BNP_value"].apply(lambda x: int(x > 400))
val_df["heart"] = val_df["BNP_value"].apply(lambda x: int(x > 400))

train_df = train_df.sort_index().merge(df_datapaths, left_index=True, right_index=True)
test_df = test_df.sort_index().merge(df_datapaths, left_index=True, right_index=True)
val_df = val_df.sort_index().merge(df_datapaths, left_index=True, right_index=True)

train_df["filepaths"] = train_df["filepaths"].str.replace("jmryan", "ddavilag")
test_df["filepaths"] = test_df["filepaths"].str.replace("jmryan", "ddavilag")
val_df["filepaths"] = val_df["filepaths"].str.replace("jmryan", "ddavilag")
train_df.shape, test_df.shape, val_df.shape

train_df.reset_index(names="unique_key", inplace=True)
val_df.reset_index(names="unique_key", inplace=True)
test_df.reset_index(names="unique_key", inplace=True)
new_valid.reset_index(names="unique_key", inplace=True)

train_df = train_df.to_numpy()
val_df = val_df.to_numpy()
test_df = test_df.to_numpy()


def run_all(df_train, df_val):
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
