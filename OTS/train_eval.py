import os
import torch
import argparse
import time
import numpy as np
import random
from PIL import Image
from torch.backends import cudnn
# from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio
from pytorch_msssim import ssim
import torch.nn as nn
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler
import torchvision.transforms as transforms
import torchvision.transforms.functional as F2
from collections import OrderedDict
from modules import *
from dataset import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#  Training function implementing the loss function from Section 3.5 ( 9)
def Train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    warmup_epochs=3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1
    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('s')
    iter_timer = Timer('s')
    best_psnr=-1
    for epoch_idx in range(epoch, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            dark_channel = dark_channel_prior(input_img)
            A = estimate_atmospheric_light(input_img, dark_channel)
            trans_map = estimate_transmission(input_img, A)
                                      
            optimizer.zero_grad()

            pred_img = model(input_img, trans_map)

            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3
            label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)
            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)
            label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)
            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)
            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)
            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3
            loss = loss_content + 0.1 * loss_fft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            optimizer.step()
            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            if (iter_idx + 1) % args.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_last_lr()[0], iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({
            'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_idx
        }, overwrite_name)
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val = Valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)

# Evaluation function for testing dehazing performance (Tables 3-7)
def Eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    factor = 32
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = F.pad(input_img, (0, padw, 0, padh), 'reflect')

            # 计算透射率图
            A = estimate_atmospheric_light(input_img, dark_channel_prior(input_img))
            trans_map = estimate_transmission(input_img, A)

            tm = time.time()
            pred = model(input_img, trans_map)[2]
            pred = pred[:,:,:h,:w]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # label_img = (label_img).cuda()
            label_img = label_img.to(device)
            psnr_val = 10 * torch.log10(1 / F.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))	
            ssim_val = ssim(F.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F2.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
            
            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))
        print("Average time: %f" % adder.average())

# Validation function (from valid.py)
def Valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ots = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for idx, data in enumerate(ots):
            input_img, label_img = data
            input_img = input_img.to(device)
            label_img = label_img.to(device) #add

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = F.pad(input_img, (0, padw, 0, padh), 'reflect')

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            # 计算透射率图
            A = estimate_atmospheric_light(input_img, dark_channel_prior(input_img))
            trans_map = estimate_transmission(input_img, A)
			
            pred = model(input_img, trans_map)[2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
