import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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
        # plt.imshow(im,cmap='gray')
        # plt.show()
        # returns image, bnpp value log, binary variable for edema

        # resnet
        return torch.load(row[4]).view(1, 224, 224).expand(3, -1, -1), row[1], row[3]

        # vgg?
        # return torch.load(row[4]).view(1, 224, 224), row[1], row[3]
