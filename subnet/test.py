# -*- coding: utf8 -*-
import os
import argparse

import pandas as pd
import torch
import cv2
import logging
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
import json
import datetime
from pytorch_msssim import ms_ssim
from tqdm import tqdm

from dataset import DataSet, UVGDataSet_I, KodakDataSet, KodakForEval, HEVCDataset, DIV2KDataset, UVGDataset
from tensorboardX import SummaryWriter
from drawuvg import uvgdrawplt
from drawkodak import kodakdrawplt
from subnet.src.models.CAM_net import *
import src.models.utils as utility
import random
from diffusers import LDMSuperResolutionPipeline
from accelerate import Accelerator
from subnet.src.lpips_pytorch import lpips
from accelerate import DistributedDataParallelKwargs

def Var(x):
    return Variable(x.cuda())


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

    input_tensor = input_tensor.to(torch.device('cpu'))

    input_tensor = input_tensor.squeeze()

    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()

    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def single_test(test_dataset_I, global_step, model, lmd):

    model.cuda().eval()
    with torch.no_grad():
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        sumlpips = 0
        train_lambda_tensor = torch.tensor(lmd)
        cnt = test_dataset_I.__len__()
        print(cnt)
        latents_dtype = next(model.unet.parameters()).dtype
        sigma = model.scheduler.init_noise_sigma
        test_dataset_I =tqdm(test_dataset_I)

        for input_image, out_path in test_dataset_I:
            input_image = torch.from_numpy(input_image).unsqueeze(0)
            frame = Var(input_image)
            height, width = frame.shape[-2:]

            latents_shape = (1, 3, height // 4, width // 4)
            latents = torch.randn(latents_shape, dtype=latents_dtype)
            latents = Var(latents * sigma)

            clipped_recon_image, _, distortion, lps_distortion, bpp = model(frame, latents, train_lambda_tensor, 2048)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_image_tensor2cv2(clipped_recon_image, out_path)

            bpp_c = torch.mean(bpp).cpu().detach().numpy()
            psnr_c = torch.mean(10 * (torch.log(1. / distortion) / np.log(10))).cpu().detach().numpy()
            msssim_c = ms_ssim(clipped_recon_image.cpu().detach(), frame.cpu().detach(), data_range=1.0,
                               size_average=True).numpy()
            lpips_c = torch.mean(lps_distortion).cpu().detach().numpy()

            sumbpp += (bpp_c)
            sumpsnr += (psnr_c)
            summsssim += (msssim_c)
            sumlpips += (lpips_c)

        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        sumlpips /= cnt
        kodakdrawplt([sumbpp], [sumpsnr], [sumlpips], global_step, testfull=True)

        print(f"lmd: {lmd}, bpp: {sumbpp}, psnr: {sumpsnr}, msssim: {summsssim}, lpips: {sumlpips}")
        return sumbpp, sumpsnr, summsssim, sumlpips


def get_argparser():
    parser_t = argparse.ArgumentParser()
    parser_t.add_argument("--pretrain", type=str, default='', help="load model",required=True)
    return parser_t


def get_model(pertrain):
    pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages",
                                                      local_files_only=True)
    scheduler = pipe.scheduler
    sigma = scheduler.init_noise_sigma
    scheduler.set_timesteps(28)
    vae = pipe.vqvae
    unet = pipe.unet
    unet.requires_grad_(False)
    model: CAM_net = CAM_net(vae, unet, scheduler)
    print("# of model parameters is: " + str(utility.count_network_parameters(model)))
    print("loading pretrain : ", pertrain)
    global_step = load_model(model, pertrain)
    return model, global_step


if __name__ == "__main__":

    opts = get_argparser().parse_args()

    model, global_step = get_model(opts.pretrain)

    div2k = DIV2KDataset("/root/DIV2K_valid_HR")
    hevcB = HEVCDataset("/root/hevcB")
    hevcE = HEVCDataset("/root/hevcE")
    hevcF = HEVCDataset("/root/hevcF")
    uvg = UVGDataset("/root/data/UVG")

    datasets = {"hevcC": hevcC, "hevcD": hevcD}
    result_df = pd.DataFrame(columns=["dataset", "model", "bpp", "psnr", "msssim", "lpips"])
    for dataset_name, dataset in datasets.items():
        test_dataset_I = dataset
        for lmd in [1, 8, 16, 32, 64, 128, 256]:
            print(f"Testing {dataset_name} with lambda {lmd}")
            test_dataset_I.set_target_folder(f"output/{dataset_name}_{lmd}")
            bpp, psnr, msssim, lpips = single_test(test_dataset_I, global_step, model, lmd=lmd)

            new_row = pd.DataFrame({"dataset": [dataset_name],
                                    "model": [opts.pretrain],
                                    "bpp": [bpp],
                                    "psnr": [psnr],
                                    "msssim": [msssim],
                                    "lpips": [lpips]})

            result_df = pd.concat([result_df, new_row], ignore_index=True)
            result_df.to_csv("result.csv")

