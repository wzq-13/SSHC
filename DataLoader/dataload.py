import os
from torch.utils.data import Dataset
import json
from collections import OrderedDict
import torch
from mmengine import fileio
import io
import numpy as np

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def opendata(path):
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff, allow_pickle=True)
    return npz_data


class My_Dataset(Dataset):
    def __init__(self, data_dir, length):
        self.data_dir = data_dir
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        file_name = f'{idx}.npz'
        file_path = os.path.join(self.data_dir, file_name)
        item = opendata(file_path)
        distance_map = item['distance_map']
        obstacles_vertices = item['obstacles_vertices']
        if not obstacles_vertices.shape == (8, 4, 2):
            print('Error obstacles shape=', obstacles_vertices.shape, 'index=', idx)
        data = {
            'distance_map': torch.tensor(distance_map, dtype=torch.float32),
            'obstacles_vertices': torch.tensor(obstacles_vertices, dtype=torch.float32),
            'target': torch.tensor(item['target'], dtype=torch.float32)
        }
        return data

def collate_fn(batch):

    collated = {
        'distance_map': torch.stack([item['distance_map'] for item in batch]),
        'obstacles_vertices': torch.stack([item['obstacles_vertices'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch])
    }

    return collated