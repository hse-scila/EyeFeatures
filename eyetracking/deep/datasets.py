import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from eyetracking.features.complex import get_heatmaps


class HeatMapDatasetTorch(Dataset):
    def __init__(self, X: pd.DataFrame, y, pk, k, transforms=None):

        self.pmk = pk
        self.size = k
        self.X, self.y = (
            get_heatmaps(X, x="norm_pos_x", y="norm_pos_y", pk=pk, k=k)[
                np.newaxis, ...
            ],
            y,
        )
        self.transforms = transforms

    def __len__(self):
        return self.X.shape[2]

    def __getitem__(self, idx):

        if self.transforms is None:
            X = self.X[:, idx, :, :]
            label = self.y.iloc[idx]

        return {
            "x": torch.tensor(X, dtype=torch.float),
            "y": torch.tensor(label, dtype=torch.long),
        }


class HeatMapDatasetLightning(pl.LightningDataModule):
    def __init__(self, X, y, label_name, pk, k, test_size, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.k = k
        self.X = X
        self.y = y
        self.label_name = label_name
        self.pk = pk
        self.test_size = test_size

    def setup(self, stage=None):

        X_train, y_train, X_val, y_val = self.split_train_val()
        self.train_dataset = HeatMapDatasetTorch(X_train, y_train, pk=self.pk, k=self.k)
        self.validation_dataset = HeatMapDatasetTorch(
            X_val, y_val, pk=self.pk, k=self.k
        )

    def train_dataloader(self):

        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        return train_loader

    def val_dataloader(self):

        valid_loader = DataLoader(
            self.validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        return valid_loader

    def split_train_val(self):

        groups_train, groups_val = train_test_split(
            self.X[self.pk].drop_duplicates().values, test_size=self.test_size
        )

        X_train = pd.concat(
            [
                self.X[pd.DataFrame(self.X[self.pk] == gr).all(axis=1)]
                for gr in groups_train
            ]
        )
        y_train = pd.concat(
            [
                self.y[pd.DataFrame(self.y[self.pk] == gr).all(axis=1)][self.label_name]
                for gr in groups_train
            ]
        )
        X_val = pd.concat(
            [
                self.X[pd.DataFrame(self.X[self.pk] == gr).all(axis=1)]
                for gr in groups_val
            ]
        )
        y_val = pd.concat(
            [
                self.y[pd.DataFrame(self.y[self.pk] == gr).all(axis=1)][self.label_name]
                for gr in groups_val
            ]
        )

        return X_train, y_train, X_val, y_val
