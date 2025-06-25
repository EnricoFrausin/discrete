from __future__ import print_function
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import ast

class Dataset_HFM(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        #self.data = pd.read_csv(csv_file)
        data = torch.load(csv_file)
        self.examples = data['examples']
        self.energies = data['energies']
        self.root_dir = root_dir
        self.transform = transform
        #self.data['examples'] = self.data['examples'].apply(lambda x: torch.tensor(ast.literal_eval(x), dtype=torch.float32))
        #self.data['energies'] = self.data['energies'].apply(lambda x: torch.tensor(int(x), dtype=torch.int32))



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        example = self.examples[idx]
        energy = self.energies[idx]
        #example = self.data.iloc[idx]['examples']
        #energy = self.data.iloc[idx]['energies']
        return example, energy
    


class Dataset_pureHFM(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        #self.data = pd.read_csv(csv_file)
        data = torch.load(csv_file)
        self.examples = data['examples']
        self.energies = data['energies']
        self.root_dir = root_dir
        self.transform = transform
        #self.data['examples'] = self.data['examples'].apply(lambda x: torch.tensor(np.fromstring(x.strip("[]"), sep=' '), dtype=torch.float32))
        #self.data['energies'] = self.data['energies'].apply(lambda x: torch.tensor(int(x), dtype=torch.int32))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        example = self.examples[idx]
        energy = self.energies[idx]
        #example = self.data.iloc[idx]['examples']
        #energy = self.data.iloc[idx]['energies']
        return example, energy
    


# https://github.com/genyrosk/pytorch-VAE-models/blob/master/dsprites/data_dsprites.py

class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, filepath, transform=None):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.filepath = f'{self.dir}/{self.filename}'
        dataset_zip = np.load(filepath, allow_pickle=True, encoding='bytes')

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        value = self.latents_values[idx].astype(np.float32)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        if self.transform:
            sample = self.transform(sample)
        return sample, value


def load_dsprites(filepath='/Users/enricofrausin/Programmazione/PythonProjects/Fisica/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                train_split=0.9, shuffle=True, seed=42, batch_size=64, reduce_dataset = True):
    # img_size = 64
    # path = os.path.join(dir, 'dsprites-dataset')

    dataset = DisentangledSpritesDataset(filepath, transform=transforms.ToTensor())



    # Create data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    if reduce_dataset:
        max_train = 60000
        max_val = 10000
        train_indices = train_indices[:max_train]
        val_indices = val_indices[:max_val]
    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader