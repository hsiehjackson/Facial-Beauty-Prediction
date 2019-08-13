# -*- coding: utf-8

import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]))
        image = Image.open(img_name)
        annotations = self.annotations.iloc[idx, 1:].values
        annotations = annotations/60
        annotations = annotations.astype('float').reshape(-1, 1)
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
