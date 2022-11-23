import torch
import math
import torch.nn as nn
import pandas as pd
import os, time, sys
import numpy as np
import pytorch_lightning as pl
import h5py
sys.path.append(os.path.dirname(os.path.realpath('.')))
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from PIL import Image
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
import glob
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = 'svg'
pio.templates.default = 'plotly_white'
from helpers.lightning_interface import *
from helpers.heart_dataset import PreprocessedImageDataset
from helpers.supernet import SuperNet
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

TEST_PATH = '/home/jmryan/private/DSC180/A/test/testdata.csv'
TRAIN_PATH = '/home/jmryan/private/DSC180/A/train/traindata.csv'
VAL_PATH = '/home/jmryan/private/DSC180/A/val/valdata.csv'


def run_all(df_train, df_val, BATCH_SIZE):
    print(BATCH_SIZE)
    train_dataset = PreprocessedImageDataset(df=df_train.to_numpy())
    val_dataset = PreprocessedImageDataset(df=df_val.to_numpy())
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers = 0, shuffle=False)
    
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    
    if torch.cuda.is_available():
        dev = 'gpu'
    else:
        dev = 'cpu'
    
    net = SuperNet(layer_defs=None, is_transfer=True, model = model, lr_scheduler=True, lr = 1e-5, batch_size=BATCH_SIZE)
    trainer = pl.Trainer(accelerator= dev, max_epochs=100, callbacks=[EarlyStopping(monitor="val_auc", mode="max")],
                         enable_progress_bar=False, logger=False, enable_checkpointing=False)
    net.train()
    trainer.fit(net, train_dl, val_dl)
    
    plt.plot(np.arange(len(net.val_loss_epoch) - 1),  net.val_loss_epoch[1:])
    plt.plot(np.arange(len(net.train_loss_epoch)),  net.train_loss_epoch)
    plt.legend(['Val','Train'])
    plt.show()

def main():
    args = sys.argv[1:]
    if args[0] == 'test':
        df_train, df_val = train_test_split(pd.read_csv(TEST_PATH, index_col=0), test_size = 0.2)
    elif args[0] == 'train':
        df_train = pd.read_csv(TRAIN_PATH, index_col=0)
        df_val = pd.read_csv(VAL_PATH, index_col=0)
    
    BATCH_SIZE = eval(args[1])
    run_all(df_train, df_val, BATCH_SIZE)
    
if __name__ == '__main__':
    main()
