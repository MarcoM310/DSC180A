import torch
import h5py
import numpy as np
import numpy as np
import pandas as pd
import h5py
import torch
from torchvision.transforms import functional as Func
from torchvision import transforms as T
from PIL import Image
import os

HF_PATH = "/home/ddavilag/teams/dsc-180a---a14-[88137]/bnpp_frontalonly_1024_"
SAVE_PATH = "/home/ddavilag/teams/dsc-180a---a14-[88137]/bnpp_224_pandas/"


def read_ins():
    hfs = []
    print(os.path.exists(hf_path + str(10) + ".hdf5"))
    hfs.append(h5py.File(HF_PATH + str(10) + ".hdf5", "r"))
    cols = [
        "unique_key",
        "bnpp_value_log",
        "BNPP_weight",
        "PNA_mask",
        "PNA_wight_mask",
        "BNP_value",
        "age_at_sampletime",
    ]
    test_df = pd.read_csv("bnpp_test.csv", usecols=cols).set_index("unique_key")
    train_df = pd.read_csv("bnpp_train.csv", usecols=cols).set_index("unique_key")
    val_df = pd.read_csv("bnpp_val.csv", usecols=cols).set_index("unique_key")
    train_df["heart"] = train_df["BNP_value"].apply(lambda x: int(x > 400))
    test_df["heart"] = test_df["BNP_value"].apply(lambda x: int(x > 400))
    val_df["heart"] = val_df["BNP_value"].apply(lambda x: int(x > 400))
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
    keys = []
    file_paths = []
    i = 0
    for hf in hfs:
        for key in list(hf.keys()):
            i += 1
            #             im = hf[key][:,:]
            #             im = change_im(im)
            folder_path = SAVE_PATH + str(key) + "/"
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            file_path = folder_path + f"{key}_224.pandas"
            #             torch.save(im, file_path)
            keys.append(key)
            file_paths.append(file_path)
            if i % 500 == 0:
                print(i)
        np.array(file_paths).tofile(
            "/home/ddavilag/teams/dsc-180a---a14-[88137]/df_bnpp_datapaths.csv", sep=","
        )
    np.array(keys).tofile(
        "/home/ddavilag/teams/dsc-180a---a14-[88137]/df_bnpp_keys.csv", sep=","
    )


pd.DataFrame({"filepath": file_paths, "keys": keys}).to_pickle(
    "/home/ddavilag/teams/dsc-180a---a14-[88137]/df_bnpp_datapaths.pandas"
)
save_files()
