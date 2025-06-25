import os
import torch
import argparse
import time
import numpy as np
import random
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2
from collections import OrderedDict
from modules import *
from train_eval import *


#  Spatial Context Module in SPM for feature extraction at different scales
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        # Defines a sequential stack of convolutional layers to extract spatial features
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),  # Initial conv reduces channels
            BasicConv(out_plane//4, out_plane//2, kernel_size=1, stride=1, relu=True),  # 1x1 conv for channel expansion
            BasicConv(out_plane//2, out_plane//2, kernel_size=3, stride=1, relu=True),  # 3x3 conv for spatial context
            BasicConv(out_plane//2, out_plane, kernel_size=1, stride=1, relu=False),  # Final 1x1 conv to target channels
            nn.InstanceNorm2d(out_plane, affine=True)  # Normalizes features across spatial dimensions
        )
    
    def forward(self, x):
        # Applies the convolutional stack to the input to extract multi-scale spatial features
        x = self.main(x)
        return x

# Fast Fourier Transform function in FPM ( 4)
def FFT(x):
    # Performs 2D FFT on the input tensor along the last two dimensions (height, width)
    x = torch.fft.fft2(x, dim=(-2, -1))
    # Computes the amplitude (magnitude) of the frequency components
    amp = torch.abs(x)
    # Computes the phase (angle) of the frequency components
    pha = torch.angle(x)
    return amp, pha

# Inverse Fast Fourier Transform function in FPM ( 4)
def IFFT(amp, pha):
    # Reconstructs real part using amplitude and cosine of phase
    real = amp * torch.cos(pha)
    # Reconstructs imaginary part using amplitude and sine of phase
    imag = amp * torch.sin(pha)
    # Combines real and imaginary parts into a complex tensor and applies inverse FFT
    return torch.fft.ifft2(torch.complex(real, imag), dim=(-2, -1))

# Amplitude and phase processing in FPM ( 2-3)
class Fre_AP(nn.Module):
    def __init__(self, channel, num_res=1) -> None:
        super().__init__()
        self.conv_amp = BasicConv(channel, channel, kernel_size=1, stride=1)  # 幅度卷积
        self.conv_pha = BasicConv(channel, channel, kernel_size=1, stride=1)  # 相位卷积

    def forward(self, x):
        x1_amp, x1_pha = FFT(x)  # 分解为幅度和相位（假设 FFT 函数已定义）
        x1_amp = self.conv_amp(x1_amp)  # 处理幅度
        x1_pha = self.conv_pha(x1_pha)  # 处理相位
        x = torch.abs(IFFT(x1_amp, x1_pha))  # 逆变换并取幅度（假设 IFFT 函数已定义）
        return x
   
# Real and imaginary component processing in FPM ( 1)
# Real, Imaginary, Amplitude, Phase: 频域特征处理
# 对应 Fre_RI 和 Fre_AP 类
class Fre_RI(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.conv_real = BasicConv(channel, channel, kernel_size=1, stride=1)  # 实部卷积
        self.conv_imag = BasicConv(channel, channel, kernel_size=1, stride=1)  # 虚部卷积

    def forward(self, x):
        x = torch.fft.fft2(x, norm='backward')  # 傅里叶变换
        x_real = x.real  # 提取实部
        x_imag = x.imag  # 提取虚部
        x_real = self.conv_real(x_real)  # 处理实部
        x_imag = self.conv_imag(x_imag)  # 处理虚部
        x = torch.complex(x_real, x_imag)  # 重构复数张量
        x = torch.fft.ifft2(x, dim=(-2,-1), norm='backward')  # 逆傅里叶变换
        return torch.abs(x)  # 返回幅度

# Frequency branch in FPM combining real/imaginary and amplitude/phase processing
class ConvFreBranch(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # Module to process amplitude and phase in frequency domain
        self.fre_ap = Fre_AP(channel)
        # Module to process real and imaginary parts in frequency domain
        self.fre_ri = Fre_RI(channel)
    
    def forward(self, x):
        # First processes real and imaginary parts
        out = self.fre_ri(x)
        # Then processes amplitude and phase on the result
        out = self.fre_ap(out)
        return out

# Feature Aggregation Module in DIM for combining features from spatial and frequency branches
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        # Convolutional layer to merge concatenated features from two domains
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)
    
    def forward(self, x1, x2):
        # Concatenates spatial (x1) and frequency (x2) features along the channel dimension
        # Applies convolution to fuse them into a single feature map
        return self.merge(torch.cat([x1, x2], dim=1))

# DIM (Dual-Domain Interaction Module): 双域交互模块，融合空间域和频率域特征
# 对应 SpaFre 类
class SpaFre(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.spatial_scale = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),  # 空间域卷积
        )
        self.fre_scale = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),  # 频率域卷积
        )

    def forward(self, spa, fre):
        fre = self.spatial_scale(spa) + fre  # 空间域特征增强频率域
        spa = self.fre_scale(fre) + spa  # 频率域特征增强空间域
        return spa, fre
    
#  HazeTrendNet architecture integrating FPM, SPM, and DIM 
class HazeTrendNet(nn.Module):
    def __init__(self, num_res=8):
        super(HazeTrendNet, self).__init__()
        base_channel = 32  # Base number of channels for feature maps

        # 频率分支和空间分支的交互模块
        self.inter = nn.ModuleList([
            SpaFre(base_channel), SpaFre(base_channel*2), SpaFre(base_channel*4),
            SpaFre(base_channel*4), SpaFre(base_channel*2), SpaFre(base_channel)
        ])

        # 频率域编码器
        self.FreEncoder = nn.ModuleList([
            ConvFreBranch(base_channel), ConvFreBranch(base_channel*2), ConvFreBranch(base_channel*4)
        ])
        
         # 频率域解码器
        self.FreDecoder = nn.ModuleList([
            ConvFreBranch(base_channel*4), ConvFreBranch(base_channel*2), ConvFreBranch(base_channel)
        ])

        # 空间域编码器
        self.Encoder = nn.ModuleList([
            EBlock1(base_channel, num_res), EBlock1(base_channel*2, num_res), EBlock1(base_channel*4, num_res),
        ])

        # 空间域特征提取层 (downsampling and upsampling)
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),  # Input to base channels
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),  # Downsample 1
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),  # Downsample 2
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),  # Upsample 1
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),  # Upsample 2
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)  # Output to 3 channels
        ])

        # 频率域特征提取层 (mirrors spatial domain)
        self.fre_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        ## 空间域解码器 for image reconstruction
        self.Decoder = nn.ModuleList([
            DBlock1(base_channel * 4, num_res), DBlock1(base_channel * 2, num_res), DBlock1(base_channel, num_res)
        ])

        # 空间域特征融合卷积 in spatial domain
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # 频率域特征融合卷积
        self.FreConvs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        # 输出卷积 produce intermediate results
        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),  # Output at 64x64
            BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),  # Output at 128x128
        ])

        # 特征聚合模块 Feature aggregation modules for multi-scale fusion
        self.FAM1 = FAM(base_channel * 4)  # For 64x64 scale
        self.SCM1 = SCM(base_channel * 4)  # Spatial context at 64x64
        self.FAM2 = FAM(base_channel * 2)  # For 128x128 scale
        self.SCM2 = SCM(base_channel * 2)  # Spatial context at 128x128

        # 动态卷积和注意力模块
        self.dyn_conv = DynamicConv(3, base_channel * 4)
        self.attn = AttentionModule(base_channel * 4)

    def forward(self, x, trans_map):
        # 计算自适应权重
        w = compute_weight(trans_map)

        # 创建多尺度输入
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_4 = F.interpolate(x_2, scale_factor=0.5, mode='bilinear', align_corners=False)
        z2 = self.SCM2(x_2)  # Extracts spatial context at 128x128
        z4 = self.SCM1(x_4)  # Extracts spatial context at 64x64

        outputs = []

        # Spatial domain: 256x256
        x_ = self.feat_extract[0](x)  # Initial feature extraction
        res1 = self.Encoder[0](x_)  # Encodes features at 256x256
        # Frequency domain: 256x256
        x_fre = self.fre_extract[0](x)  # Initial frequency feature extraction
        fre_res1 = self.FreEncoder[0](x_fre)  # Encodes frequency features
        res1, fre_res1 = self.inter[0](res1, fre_res1)  # Fuses spatial and frequency features

        # Spatial domain: 128x128
        z = self.feat_extract[1](res1)  # Downsamples to 128x128
        z = self.FAM2(z, z2)  # Fuses with multi-scale context
        res2 = self.Encoder[1](z)  # Encodes features at 128x128
        # Frequency domain: 128x128
        z_fre = self.fre_extract[1](fre_res1)  # Downsamples frequency features
        fre_res2 = self.FreEncoder[1](z_fre)  # Encodes frequency features
        res2, fre_res2 = self.inter[1](res2, fre_res2)  # Fuses features

        # Spatial domain: 64x64 (bottleneck)
        z = self.feat_extract[2](res2)  # Downsamples to 64x64
        z = self.FAM1(z, z4)  # Fuses with multi-scale context
        z = self.Encoder[2](z)  # Encodes features at 64x64
        # Frequency domain: 64x64
        z_fre = self.fre_extract[2](fre_res2)  # Downsamples frequency features
        z_fre = self.FreEncoder[2](z_fre)  # Encodes frequency features
        z, z_fre = self.inter[2](z, z_fre)  # Fuses features at bottleneck

        # Decoding starts
        z = self.Decoder[0](z)  # Decodes spatial features at 64x64
        z_fre = self.FreDecoder[0](z_fre)  # Decodes frequency features at 64x64
        z_ = self.ConvsOut[0](z)  # Produces intermediate output at 64x64
        z, z_fre = self.inter[3](z, z_fre)  # Fuses features

        # Spatial domain: Upsample to 128x128
        z = self.feat_extract[3](z)  # Upsamples to 128x128
        outputs.append(z_ + x_4)  # Adds residual connection and stores output
        z = torch.cat([z, res2], dim=1)  # Concatenates with skip connection
        z = self.Convs[0](z)  # Fuses features
        # Frequency domain: 128x128
        z_fre = self.fre_extract[3](z_fre)  # Upsamples frequency features
        z_fre = torch.cat((z_fre, fre_res2), dim=1)  # Concatenates with skip connection
        z_fre = self.FreConvs[0](z_fre)  # Fuses frequency features

        z = self.Decoder[1](z)  # Decodes to 128x128
        z_fre = self.FreDecoder[1](z_fre)  # Decodes frequency features
        z_ = self.ConvsOut[1](z)  # Produces intermediate output at 128x128
        z, z_fre = self.inter[4](z, z_fre)  # Fuses features

        # Spatial domain: Upsample to 256x256
        z = self.feat_extract[4](z)  # Upsamples to 256x256
        outputs.append(z_ + x_2)  # Adds residual connection and stores output
        z = torch.cat([z, res1], dim=1)  # Concatenates with skip connection
        z = self.Convs[1](z)  # Fuses features
        # Frequency domain: 256x256
        z_fre = self.fre_extract[4](z_fre)  # Upsamples frequency features
        z_fre = torch.cat((z_fre, fre_res1), dim=1)  # Concatenates with skip connection
        z_fre = self.FreConvs[1](z_fre)  # Fuses frequency features

        z = self.Decoder[2](z)  # Final decoding to 256x256
        z_fre = self.FreDecoder[2](z_fre)  # Final frequency decoding
        z, z_fre = self.inter[5](z, z_fre)  # Final fusion

        z = self.feat_extract[5](z)  # Final spatial output layer
        z_fre = self.fre_extract[5](z_fre)  # Final frequency output layer
        outputs.append(z + x + z_fre)  # Combines spatial, input, and frequency for final output

        # 在空间处理中使用动态卷积
        # print("z shape before dyn_conv:", z.shape)  # 应输出 [batch_size, 3, H, W]
        z = self.dyn_conv(z, w)
        # 在双域交互中使用注意力机制
        z = self.attn(z, trans_map)

        return outputs  # Returns list of multi-scale outputs

def build_net():
    return HazeTrendNet()

# Main function (from main.py)
def main(args):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    model = build_net()
    print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        Train(model, args)
    elif args.mode == 'test':
        Eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='HazeTrendNet', type=str)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='/SOTS/indoor')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--valid_freq', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--test_model', type=str, default='its.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', 'HazeTrendNet', 'its/')
    args.result_dir = os.path.join('results/', args.model_name, 'test')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    print(args)
    main(args)
