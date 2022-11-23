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

from src.models.train_model import train1Epoch
from src.models.predict_model import test1Epoch
from src.data import make_dataset
from src.features.build_features import files2df, PreprocessedImageDataset
from src.helper.models import VGG

torch.cuda.empty_cache()
import seaborn as sns

test_path = "/home/ddavilag/private/DSC180A_Final/DSC180A/test/testdata.csv"
# test_path = "/home/ddavilag/private/DSC180A_Final/DSC180A/data/out/testdata.csv"
train_path = "/home/ddavilag/private/DSC180A_Final/DSC180A/data/out/traindata.csv"
val_path = "/home/ddavilag/private/DSC180A_Final/DSC180A/data/out/valdata.csv"


def run_all(df_train, df_val):
    train_set = PreprocessedImageDataset(df=df_train)
    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    valid_set = PreprocessedImageDataset(df=df_val)
    valid_loader = DataLoader(
        valid_set,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    if torch.cuda.is_available():
        dev = "gpu"
    else:
        dev = "cpu"

    model = VGG("VGG16").to(device)

    loss_fn = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epoch_number = 0

    EPOCHS = 15

    # the scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    tlosses, vlosses = np.array([]), np.array([])

    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))

        for param in model.parameters():
            param.requires_grad = True
        avg_tloss = train1Epoch(
            epoch_number, model, optimizer, loss_fn, train_loader
        )  # , writer)

        for param in model.parameters():
            param.requires_grad = False
        with torch.no_grad():
            avg_vloss = test1Epoch(epoch_number, model, loss_fn, valid_loader)

        print("LOSS train {} valid {}".format(avg_tloss, avg_vloss))

        tlosses = np.append(tlosses, avg_tloss)
        vlosses = np.append(vlosses, avg_vloss)
        print(tlosses)

        epoch_number += 1
        scheduler.step(avg_vloss)

    epochs = np.arange(1, EPOCHS + 1)
    df = pd.DataFrame(data={"train loss": tlosses, "valid loss": vlosses})
    sns.set(style="whitegrid")
    g = sns.FacetGrid(df, height=6)
    g = g.map(sns.lineplot, x=epochs, y=tlosses, marker="o", label="train")
    g = g.map(sns.lineplot, x=epochs, y=vlosses, color="red", marker="o", label="valid")
    g.set(ylim=(0, None))
    g.add_legend()
    plt.xticks(epochs)
    plt.show()


def main(targets):
    if targets[0] == "test":
        df_train, df_val = train_test_split(
            pd.read_csv(test_path, index_col=0), test_size=0.2
        )
    elif targets[0] == "train":
        df_train = pd.read_csv(train_path, index_col=0)
        df_val = pd.read_csv(val_path, index_col=0)

    run_all(df_train, df_val)


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)
