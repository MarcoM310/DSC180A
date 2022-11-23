import h5py
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from PIL import Image
import os
from tqdm import tqdm

HF_PATH = "/home/ddavilag/teams/dsc-180a---a14-[88137]/bnpp_frontalonly_1024_"
SAVE_PATH = "/home/ddavilag/private/data/bnpp_224_pandas/"


def read_ins():
    hfs = []
    for i in range(11):
        if i >= 7 and i != 10:
            continue
        # print(os.path.exists(HF_PATH + str(i) + ".hdf5"))
        hfs.append(h5py.File(HF_PATH + str(i) + ".hdf5", "r"))
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
    return hfs, train_df, test_df, val_df


def change_im(im):
    pil = T.ToPILImage()
    tens = T.ToTensor()
    resize = T.Resize([224, 224], interpolation=T.InterpolationMode.BILINEAR)
    im = (im - im.min()) / (im.max() - im.min())
    im = tens(resize(pil(im.copy())))[0]
    return im


def save_files():
    hfs, train_df, test_df, val_df = read_ins()
    keys, file_paths = [], []

    for hf in hfs:
        for i, key in tqdm(enumerate(list(hf.keys())), total=len(list(hf.keys()))):
            im = hf[key][:, :]
            im = change_im(im)
            folder_path = SAVE_PATH + str(key) + "/"
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            file_path = folder_path + f"{key}_224.pandas"
            torch.save(im, file_path)
            keys.append(key)
            file_paths.append(file_path)
        np.array(file_paths).tofile(
            "/home/ddavilag/private/data/df_bnpp_datapaths.csv",
            sep=",",
        )
    np.array(keys).tofile("/home/ddavilag/private/data/df_bnpp_keys.csv", sep=",")
    pd.DataFrame({"filepath": file_paths, "keys": keys}).to_pickle(
        "/home/ddavilag/private/data/df_bnpp_datapaths.pandas"
    )
