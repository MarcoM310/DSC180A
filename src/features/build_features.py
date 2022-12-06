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

BATCH_SIZE = 1
NUM_WORKERS = 1
PIN_MEMORY = True


def files2df(threshold=400):
    keys = np.genfromtxt(
        "/home/ddavilag/private/data/df_bnpp_keys.csv", delimiter=",", dtype=str
    )
    file_paths = np.genfromtxt(
        "/home/ddavilag/private/data/df_bnpp_datapaths.csv", delimiter=",", dtype=str
    )
    df = pd.DataFrame({"key": keys, "path": file_paths})
    df.key = df.key.apply(lambda x: eval(x))
    df.path = df.path.apply(lambda x: eval(x))
    df.set_index(keys="key", inplace=True)

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

    train_df = train_df.sort_index().merge(df, left_index=True, right_index=True)
    test_df = test_df.sort_index().merge(df, left_index=True, right_index=True)
    val_df = val_df.sort_index().merge(df, left_index=True, right_index=True)

    train_df.reset_index(names="unique_key", inplace=True)
    val_df.reset_index(names="unique_key", inplace=True)
    test_df.reset_index(names="unique_key", inplace=True)

    train_df["heart"] = train_df["BNP_value"].apply(lambda x: int(x > threshold))
    test_df["heart"] = test_df["BNP_value"].apply(lambda x: int(x > threshold))
    val_df["heart"] = val_df["BNP_value"].apply(lambda x: int(x > threshold))

    return train_df, test_df, val_df


class PreprocessedImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx, :]
        # returns image, bnpp value log, binary variable for edema

        return torch.load(row[3]).view(1, 224, 224).expand(3, -1, -1), row[1], row[3]


def Loader(dataset, mode):
    if mode == "train":
        shuffle = True
    else:
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
