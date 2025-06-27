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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


import random
class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None):
        self.image_dir = image_dir
        hazy_dir = os.path.join(image_dir, 'hazy')
        gt_dir = os.path.join(image_dir, 'gt')
        self.image_list = [f for f in os.listdir(hazy_dir) 
                           if os.path.isfile(os.path.join(hazy_dir, f)) 
                           and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self._check_image(self.image_list)
        self.image_list.sort()
        self.gt_files = {}
        for hazy_name in self.image_list:
            prefix = hazy_name.split('_')[0]
            gt_files = [f for f in os.listdir(gt_dir) if f.startswith(prefix + '.')]
            if not gt_files:
                raise FileNotFoundError(f"No ground truth file found for {hazy_name}")
            self.gt_files[hazy_name] = gt_files[0]
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        hazy_name = self.image_list[idx]
        gt_name = self.gt_files[hazy_name]
        image = Image.open(os.path.join(self.image_dir, 'hazy', hazy_name)).convert('RGB')
        label = Image.open(os.path.join(self.image_dir, 'gt', gt_name)).convert('RGB')
        ps = self.ps

        if self.ps is not None:
            image = F2.to_tensor(image)
            label = F2.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]

            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)
            
            image = image[:, rr:rr+ps, cc:cc+ps]
            label = label[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:
                image = image.flip(2)
                label = label.flip(2)
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
            if len(splits) < 2 or splits[-1] not in ['png', 'jpg', 'jpeg']:
                print(f"Invalid file: {x}")
                raise ValueError(f"File {x} does not have a valid image extension.")
            
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
def train_dataloader(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader