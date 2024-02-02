# -*- coding: utf8 -*-
import os
import argparse
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
from dataset import DataSet, UVGDataSet_I, KodakDataSet
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



torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
num_workers = gpu_num*4
print('gpu_num:', gpu_num)
cur_lr = base_lr = 0.0001  # * gpu_num
train_lambda = 2048
print_step = 100
cal_step = 10
# print_step = 10
warmup_step = 0  # // gpu_num
gpu_per_batch = 2
test_step = 10000  # // gpu_num
tot_epoch = 100
tot_step = 3000000
decay_interval = 1800000
lr_decay = 0.1
logger = logging.getLogger("VideoCompression")
tb_logger = None
global_step = 0
ifarr = 0
ifout = 0
recon_path = 'recon/recon.bin'


##################################################################################
pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages", local_files_only = True)
scheduler = pipe.scheduler
sigma = scheduler.init_noise_sigma
scheduler.set_timesteps(28)
vae = pipe.vqvae
unet = pipe.unet
latents_dtype = next(unet.parameters()).dtype
unet.requires_grad_(False)
# vae.requires_grad_(False)
##################################################################################

once_strings = []
def print_once(strings):
    if strings in once_strings:
        return
    print(strings)
    once_strings.append(strings)
print_once("=== main ===")

# def geti(lamb):
#     if lamb == 2048:
#         return 'H265L20'
#     elif lamb == 1024:
#         return 'H265L23'
#     elif lamb == 512:
#         return 'H265L26'
#     elif lamb == 256:
#         return 'H265L29'
#     else:
#         print("cannot find lambda : %d"%(lamb))
#         exit(0)

# ref_i_dir = geti(train_lambda)


parser = argparse.ArgumentParser(description='FVC reimplement')

parser.add_argument('-l', '--log', default='',
                    help='output training details')
parser.add_argument('-p', '--pretrain', default='',
                    help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--testuvg', action='store_true')
parser.add_argument('--testvtl', action='store_true')
parser.add_argument('--testmcl', action='store_true')
parser.add_argument('--testauc', action='store_true')
parser.add_argument('--rerank', action='store_true')
parser.add_argument('--allpick', action='store_true')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of Reid in json format')


def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, test_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, ref_i_dir, ifend, iftest, ifmsssim, msssim_lambda, ifout
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'test_step' in config:
        test_step = config['test_step']
        print('teststep : ', test_step)
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        # ref_i_dir = geti(train_lambda)
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']
    if 'ifend' in config:
        ifend = config['ifend']
    if 'iftest' in config:
        iftest = config['iftest']
    if 'ifmsssim' in config:
        ifmsssim = config['ifmsssim']
    if 'msssim_lambda' in config:
        msssim_lambda = config['msssim_lambda']
    if 'ifout' in config:
        ifout = config['ifout']

def Var(x):
    return Variable(x.cuda())


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    
    input_tensor = input_tensor.to(torch.device('cpu'))

    input_tensor = input_tensor.squeeze()
    
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)


def testuvg(global_step):
    with torch.no_grad():
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)
        net.eval()

        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        train_lambda_tensor = Var(torch.tensor(train_lambda).view(1))
        train_lambda_tensor_boudary = Var(torch.tensor(2048).view(1))

        for batch_idx, input in enumerate(test_loader):

            log = 'testing squeence: {} '.format(batch_idx)
            print(log)

            input_images = input
            seqlen = input_images.size()[1]
            cnt += seqlen
            # print(seqlen)

            for i in range(0,seqlen):
                image1 = input_images[:, i, :, :, :]
                # image2 = input_images[:, i+1, :, :, :]
                # image3 = input_images[:, i+2, :, :, :]
                # frame1, frame2, frame3 = Var(image1), Var(image2), Var(image3)
                frame1 = Var(image1)

                clipped_recon_image, feature_map, distortion, bpp_y, bpp_z, bpp = net(frame1, train_lambda_tensor, train_lambda_tensor_boudary)

                bpp_c = torch.mean(bpp).cpu().detach().numpy()
                psnr_c = torch.mean(10 * (torch.log(1. / distortion) / np.log(10))).cpu().detach().numpy()
                msssim_c = ms_ssim(clipped_recon_image.cpu().detach(), image1, data_range=1.0,
                                     size_average=True).numpy()
                
                sumbpp += (bpp_c)
                sumpsnr += (psnr_c)
                summsssim += (msssim_c)
        
            log = 'bpp: {:.4f}  psnr: {:.4f} ssim: {:.4f}'.format(sumbpp / cnt, sumpsnr / cnt, summsssim / cnt)
            print(log)
                

        log = "global step %d : " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (
            sumbpp, sumpsnr, summsssim)
        logger.info(log)
        uvgdrawplt([sumbpp], [sumpsnr], [summsssim], global_step, testfull=True)

def testkodak(global_step):
    test_loader = DataLoader(dataset=test_dataset_I, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)
    net.cuda().eval()
    with torch.no_grad():
        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        sumlpips = 0
        train_lambda_tensor = torch.tensor(train_lambda)
        cnt = test_loader.__len__()
        print(cnt)

        for num, input in enumerate(test_loader):
            frame = Var(input)
            height, width = frame.shape[-2:]

            latents_shape = (1, 3, height // 4, width // 4)
            latents = torch.randn(latents_shape, dtype=latents_dtype)
            latents = Var(latents * sigma)

            clipped_recon_image, distortion, lps_distortion, bpp, feature_distortion  = net(frame, latents, train_lambda_tensor, 2048)
            recon_path = "./fullpreformance/kodak_recon/"
            img_name = 'kodim' + str(num+1).zfill(2) + '_' + str(train_lambda) + '_recon.png'
            save_image_tensor2cv2(clipped_recon_image, os.path.join(recon_path, img_name))

            bpp_c = torch.mean(bpp).cpu().detach().numpy()
            psnr_c = torch.mean(10 * (torch.log(1. / distortion) / np.log(10))).cpu().detach().numpy()
            msssim_c = ms_ssim(clipped_recon_image.cpu().detach(), frame.cpu().detach(), data_range=1.0,
                                    size_average=True).numpy()
            lpips_c = torch.mean(lps_distortion).cpu().detach().numpy()
            
            sumbpp += (bpp_c)
            sumpsnr += (psnr_c)
            summsssim += (msssim_c)
            sumlpips += (lpips_c)
                
        log = "global step %d : " % (global_step) + "\n"
        logger.info(log)
        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        sumlpips /= cnt
        log = "Kodakdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n, average lpips: %.6lf\n" % (
            sumbpp, sumpsnr, summsssim, sumlpips)
        logger.info(log)
        kodakdrawplt([sumbpp], [sumpsnr], [sumlpips], global_step, testfull=True)

def train(epoch, global_step):
    print("epoch", epoch)
    global gpu_per_batch
    global optimizer
    global cur_lr 
    global net 
    global scheduler 

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=num_workers, batch_size=gpu_per_batch,
                              pin_memory=True)
    train_loader, net, optimizer, scheduler = accelerator.prepare(train_loader, net, optimizer, scheduler)

    net.train()

    
    bat_cnt = 0
    cal_cnt = 0
    sumloss = 0
    sumpsnr = 0
    sumbpp = 0
    tot_iter = len(train_loader)
    t0 = datetime.datetime.now()
    
    for batch_idx, input in enumerate(train_loader):
        global_step += 1
        bat_cnt += 1
        image1, image2, image3 = Variable(input[0]), Variable(input[1]), Variable(input[2])
        quant_noise_feature, quant_noise_z = Variable(input[3]), Variable(input[4])
        latents = Variable(input[5])
        var_lambda = random.randint(8,256)
        clipped_recon_bimage, distortion, lpips_distortion, bpp, feature_distortion = net(input_image = image2, latents = latents, lmd=var_lambda, lmd_boundary=2048, previous_frame = None, feature_frame=None, quant_noise_feature=quant_noise_feature, quant_noise_z=quant_noise_z)
        
        distortion, bpp, lpips_distortion, feature_distortion = torch.mean(distortion), torch.mean(bpp), torch.mean(lpips_distortion), torch.mean(feature_distortion)
        rd_loss = var_lambda * (distortion + 0.05 * lpips_distortion + 0.05 * feature_distortion) + bpp
        
        optimizer.zero_grad()
        accelerator.backward(rd_loss)
        
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 0.5)
        optimizer.step()

        if global_step % cal_step == 0:
            cal_cnt += 1
            if distortion > 0:
                psnr = 10 * (torch.log(1 * 1 / distortion) / np.log(10)).cpu().detach().numpy()
            else:
                psnr = 100
            
            loss_ = rd_loss.cpu().detach().numpy()

            sumloss += loss_
            sumpsnr += psnr
            sumbpp += bpp.cpu().detach()


        if (batch_idx % print_step) == 0 and bat_cnt > 1:
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', sumloss / cal_cnt, global_step)
            tb_logger.add_scalar('psnr', sumpsnr / cal_cnt, global_step)
            tb_logger.add_scalar('bpp', sumbpp / cal_cnt, global_step)
            t1 = datetime.datetime.now()
            deltatime = t1 - t0
            log = 'Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] Avgloss:{:.6f} lr:{} time:{}'.format(epoch, batch_idx,
                                                                                                   len(train_loader),
                                                                                                   100. * batch_idx / len(
                                                                                                       train_loader),
                                                                                                   sumloss / cal_cnt,
                                                                                                   cur_lr, 
                                                                                                   (deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt)
            print(log)
            log = 'details : psnr : {:.2f} bpp : {:.6f}'.format(sumpsnr / cal_cnt, sumbpp / cal_cnt)
            print(log)
            print(f"data of last iter: distortion: {distortion}, bpp: {bpp}, lpips_distortion: {lpips_distortion}, feature_distortion: {feature_distortion}")
            bat_cnt = 0
            cal_cnt = 0
            sumbpp = sumloss = sumpsnr = 0
            t0 = t1
    log = 'Train Epoch : {:02} Loss:\t {:.6f}\t lr:{}'.format(epoch, sumloss / bat_cnt, cur_lr)
    logger.info(log)
    return global_step


if __name__ == "__main__":
    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.log != '':
        filehandler = logging.FileHandler(args.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("CAM_I training")
    logger.info(open(args.config).read())
    parse_config(args.config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    model = CAM_net(vae,unet,scheduler)
    print("# of model parameters is: " + str(utility.count_network_parameters(model)))

    if args.pretrain != '':
        print("loading pretrain : ", args.pretrain)
        global_step = load_model(model, args.pretrain)

    net = model
    # net = torch.nn.DataParallel(net, list(range(gpu_num)))
    bp_parameters = net.parameters()
    optimizer = optim.Adam(bp_parameters, lr=base_lr)
    
    global train_dataset, test_dataset
    if args.testuvg:
        # print('testing UVG')
        # test_dataset = UVGDataSet_I(refdir=ref_i_dir)
        # testuvg(global_step)
        print('testing Kodak')
        test_dataset_I = KodakDataSet()
        testkodak(global_step)
        exit(0)

    tb_logger = SummaryWriter('./events')
    train_dataset = DataSet(latents_dtype, sigma, "./data/vimeo_septuplet/test.txt")
    # test_dataset = UVGDataSet_I(refdir=ref_i_dir)
    test_dataset_I = KodakDataSet()
    stepoch = global_step // (train_dataset.__len__() // (gpu_per_batch* gpu_num))  # * gpu_num))

    stage_progress_4 = [80765*7, 80765*8]
    stage_progress = len(stage_progress_4)-1
    lrs = [1e-4, 1e-5]

    

    for epoch in range(stepoch, tot_epoch):
        for i in range(len(stage_progress_4)):
            if global_step < stage_progress_4[i] - 1:
                stage_progress = i
                break 
        
        log1 = 'Processing training step: {} / 2 '.format(stage_progress+1)
        logger.info(log1)
        cur_lr = lrs[stage_progress]
            
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = cur_lr)
        if global_step > tot_step:
            save_model(model, global_step)
            print("Finish training")
            break

        global_step = train(epoch, global_step)
        save_model(model, global_step)

        if global_step > 80765*3:
            # testuvg(global_step)
            testkodak(global_step)
