# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
from PIL import Image


class ProstateDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.info = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.info.iloc[idx]['label']
        patient = self.info.iloc[idx]['patient_id']
        lesion = self.info.iloc[idx]['lesion_id']
        n_slice = self.info.iloc[idx]['slice_id']

        image_path = os.path.join(os.getcwd(),"dataset",patient,lesion,n_slice)
        image = Image.open(image_path)
        image = np.array(image)

        target = torch.tensor(0 if label == 'LG' else 1)

        return image, target, patient, lesion, n_slice


class ToTensorDataset(Subset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, transform):
        super().__init__(dataset, range(len(dataset)))
        self.transform = transform

    def __getitem__(self, idx):
        image, label, patient, lesion, n_slice = self.dataset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, patient, lesion, n_slice
