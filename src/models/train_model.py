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

LR = 0.0001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training
def train1Epoch(epoch_index, model, optimizer, loss_fn, training_loader, writer=None):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    for i, (image, bnpp, _) in tqdm(
        enumerate(training_loader), total=len(training_loader)
    ):
        # image, bnpp, _= data
        image, bnpp = image.to(device, non_blocking=True), bnpp.to(
            device, non_blocking=True
        )
        pred = model(image)
        loss = loss_fn(torch.squeeze(pred, 1), bnpp)
        # Backpropagation
        optimizer.zero_grad()  # set_to_none=True)
        loss.backward()
        optimizer.step()

        losses = np.append(losses, loss.item())
    return np.mean(losses)


def test1Epoch(epoch_index, model, loss_fn, valid_loader, tb_writer=None):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    with torch.no_grad():
        for i, (image, bnpp, _) in tqdm(
            enumerate(valid_loader), total=len(valid_loader)
        ):
            image, bnpp = image.to(device, non_blocking=True), bnpp.to(
                device, non_blocking=True
            )

            pred = model(image)
            loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
            losses = np.append(losses, loss.item())
            image.detach()
            bnpp.detach()
    return np.mean(losses)


def trainAndSave(model, train_loader, valid_loader, num_epochs=EPOCHS):
    ### trains model and saves it to "src/models/resnet.pt"
    # also checks performance on validation set each epoch
    epoch_number = 0
    loss_fn = nn.L1Loss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    tlosses, vlosses = np.array([]), np.array([])
    best_vloss = np.inf
    for epoch in range(num_epochs):
        print("EPOCH {}:".format(epoch_number + 1))

        for param in model.parameters():
            param.requires_grad = True
        avg_tloss = train1Epoch(epoch_number, model, optimizer, loss_fn, train_loader)

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
        if best_vloss > avg_vloss:
            best_vloss = avg_vloss
            torch.save(
                {
                    "epoch": epoch_number,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": tlosses[-1],
                },
                "resnet.pt",
            )
            print("model saved!")
