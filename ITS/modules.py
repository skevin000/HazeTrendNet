import os
import torch
import argparse
import time
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2
from collections import OrderedDict


#  Basic convolutional layer used throughout the network architecture (supports encoder-decoder structure)
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.PReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


#  Encoder block in the spatial branch (part of the encoder-decoder architecture)
class EBlock1(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock1, self).__init__()
        # Initializes a UNet structure to extract features from the input
        self.layers = UNet(out_channel, out_channel, num_res)
    
    def forward(self, x):
        # Passes the input through the UNet layers for feature extraction
        return self.layers(x)

#  Decoder block in the spatial branch (part of the encoder-decoder architecture)
class DBlock1(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock1, self).__init__()
        # Initializes a UNet structure to reconstruct the image from features
        self.layers = UNet(channel, channel, num_res)
    
    def forward(self, x):
        # Processes the input features through the UNet layers to rebuild the image
        return self.layers(x)

def Conv2D(in_channels, out_channels, kernel_size, padding, stride=1, has_relu=False):
    modules = OrderedDict()
    modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    if has_relu:
        modules['relu'] = nn.ReLU()
    return nn.Sequential(modules)

# Dynamic convolution kernel selection
class DynamicConv(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size_list=[3, 5]):
        super(DynamicConv, self).__init__()
        self.convs = nn.ModuleList([
            Conv2D(in_channels, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_size_list
        ])
    
    def forward(self, x, w):
        outputs = [conv(x) for conv in self.convs]
        weighted_outputs = [out * w for out in outputs]
        return sum(weighted_outputs)

# Haze density-sensitive attention mechanism
class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.conv = Conv2D(1, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, trans_map):
        trans_map = F.interpolate(trans_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        mask = self.sigmoid(self.conv(1 - trans_map))
        return x * mask    

def dark_channel_prior(img, patch_size=15):
    min_channel = torch.min(img, dim=1, keepdim=True)[0]
    kernel = torch.ones((1, 1, patch_size, patch_size), device=img.device)
    dark_channel = F.conv2d(min_channel, kernel, padding=patch_size//2)
    return dark_channel

def estimate_atmospheric_light(img, dark_channel):
    B, C, H, W = img.shape
    num_pixels = H * W
    top_pixels = int(num_pixels * 0.001)
    dark_channel_flat = dark_channel.view(B, -1)
    _, indices = torch.topk(dark_channel_flat, top_pixels, dim=1)
    A = torch.zeros(B, C, 1, 1, device=img.device)
    for b in range(B):
        for c in range(C):
            A[b, c, 0, 0] = img[b, c].view(-1)[indices[b]].max()
    return A

def estimate_transmission(img, A, omega=0.95):
    img_normalized = img / A
    dark_channel = dark_channel_prior(img_normalized)
    transmission = 1 - omega * dark_channel
    return transmission.clamp(0.1, 1.0)


def compute_weight(trans_map):
    mu = torch.mean(trans_map, dim=(2, 3), keepdim=True)
    sigma = torch.std(trans_map, dim=(2, 3), keepdim=True)
    sigma_min = 0.01
    denominator = torch.max(sigma, torch.tensor(sigma_min, device=trans_map.device))
    w = torch.sigmoid((mu - trans_map) / denominator)
    return w

def compute_gradient(img):
    grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
    grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
    grad_x = F.pad(grad_x, (0, 0, 0, 1), mode='constant', value=0)
    grad_y = F.pad(grad_y, (0, 1, 0, 0), mode='constant', value=0)
    grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return grad

#  Utility classes for training and evaluation (not directly tied to a specific module in the paper but supports overall implementation)
class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count

#  Utility class for timing operations during training and evaluation
class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider

#  Utility function to check learning rate during training
def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr

#  Residual block in SPM for multi-scale representation learning (part of the dual-path architecture)
class ResBlock1(nn.Module):
    def __init__(self, in_channel, out_channel, multi=False):
        super(ResBlock1, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),  # Base convolution
            Fine2Coarse(in_channel) if multi else nn.Identity(),  # Fine-to-Coarse processing
            Coarse2Fine(in_channel) if multi else nn.Identity(),  # Coarse-to-Fine processing
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # Output convolution
        )
        self.main1 = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            Fine2Coarse(in_channel) if multi else nn.Identity(),
            Coarse2Fine(in_channel) if multi else nn.Identity(),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2):
        x1 = self.main(x1) + x1  # Residual connection
        x2 = self.main1(x2) + x2  # Residual connection
        return x1, x2

# UNet structure in SPM for multi-scale representation learning within the spatial branch
# SPM (Spatial Processing Module): Learns multi-scale spatial features
# Corresponds to UNet class, implements SPM via multi-scale feature extraction
class UNet(nn.Module):
    def __init__(self, inchannel, outchannel, num_res) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_res-1):
            self.layers.append(ResBlock1(inchannel//2, outchannel//2))  # Add residual blocks
        self.layers.append(ResBlock1(inchannel//2, outchannel//2, multi=True))  # Last residual block supports multi-scale processing
        self.num_res = num_res
        self.down = nn.Conv2d(inchannel//2, outchannel//2, kernel_size=2, stride=2, groups=inchannel//2)  # Downsampling layer

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x1, x2 = torch.chunk(x, 2, dim=1)  # Split input into two parts for processing
            elif i == self.num_res // 4:
                x2 = self.down(x2)  # Downsample at specific layer
            elif i == self.num_res // 4 * 3:
                x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')  # Upsample to align dimensions
            x1, x2 = layer(x1, x2)  # Process through residual block
        x = torch.cat((x1, x2), dim=1)  # Concatenate results
        return x

#  Coarse-to-Fine module in SPM, part of the fine-to-coarse-to-fine strategy ( 8)
class Coarse2Fine(nn.Module):
    def __init__(self, k):
        super(Coarse2Fine, self).__init__()
        self.pools_sizes = [8, 4, 2]  # Multi-scale pooling sizes (in reverse order)
        pools, convs = [], []
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False, groups=k))  # Depthwise separable convolution
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.GELU()
        self.conv_sum = nn.Conv2d(k, k, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.convs[i](self.pools[i](x))
            else:
                y = self.convs[i](self.pools[i](x) + y_up)
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)
        return resl

# Fine-to-Coarse module in SPM, part of the fine-to-coarse-to-fine strategy
# FCF (Fine-to-Coarse-to-Fine): Feature processing strategy from fine to coarse and back to fine
class Fine2Coarse(nn.Module):
    def __init__(self, k):
        super(Fine2Coarse, self).__init__()
        self.pools_sizes = [2, 4, 8]  # Multi-scale pooling sizes
        pools, convs = [], []
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))  # Average pooling
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False, groups=k))  # Depthwise separable convolution
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.GELU()  # Activation function
        self.conv_sum = nn.Conv2d(k, k, 3, 1, 1, bias=False)  # Fusion convolution

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            if i == 0:
                y = self.convs[i](self.pools[i](x))  # Initial pooling and convolution
            else:
                y = self.convs[i](self.pools[i](x) + y_down)  # Fuse downsampled features
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.pools_sizes)-1:
                y_down = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.conv_sum(resl)
        return resl        
