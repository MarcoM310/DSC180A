import os
import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import cv2


class PreprocessedImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx, :]
        # plt.imshow(im,cmap='gray')
        # plt.show()
        # returns image, bnpp value log, binary variable for edema

        # resnet
        return torch.load(row[4]).view(1, 224, 224).expand(3, -1, -1), row[1], row[3]

        # vgg?
        # return torch.load(row[4]).view(1, 224, 224), row[1], row[3]
