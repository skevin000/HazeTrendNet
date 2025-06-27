import os
import torch
import argparse
import time
import numpy as np
import random
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

# Test data loader (supports evaluation datasets in Table 1)
def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

#  Validation data loader (supports evaluation datasets in Table 1)
def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader

#  Dataset class for dehazing (supports datasets listed in Table 1)
class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png'))
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F2.to_tensor(image)
            label = F2.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


#  Data augmentation for training (not directly tied to a specific module but supports dataset preparation)
class PairRandomCrop(transforms.RandomCrop):
    def __call__(self, image, label):
        if self.padding is not None:
            image = F2.pad(image, self.padding, self.fill, self.padding_mode)
            label = F2.pad(label, self.padding, self.fill, self.padding_mode)
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F2.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F2.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F2.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F2.pad(label, (0, self.size[0] - label.size[1]), self.fill, self.padding_mode)
        i, j, h, w = self.get_params(image, self.size)
        return F2.crop(image, i, j, h, w), F2.crop(label, i, j, h, w)


#  Composition of data augmentation transforms
class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

#  Random horizontal flip for data augmentation
class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        if random.random() < self.p:
            return F2.hflip(img), F2.hflip(label)
        return img, label

#  Conversion to tensor for data augmentation
class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        return F2.to_tensor(pic), F2.to_tensor(label)


#  Training data loader (supports dataset preparation in Table 1)
def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')
    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader