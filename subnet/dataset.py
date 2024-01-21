import os
import torch
import logging
#import cv2
from PIL import Image
import imageio
import numpy as np
import torch.utils.data as data
from os.path import join, exists
import math
import random
import sys
import json
import random
#from subnet.basics import *
#from subnet.ms_ssim_torch import ms_ssim
from pytorch_msssim import ms_ssim
#from augmentation import random_flip, random_crop_and_pad_image_and_labels
import torch.nn.functional as F

out_channel_N = 192
out_channel_M = 192
middle_channel = 7

def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

def random_crop_and_pad_image_and_labels(image1, image2, labels, size):
    combined1 = torch.cat([image1, labels], 0)
    combined2 = torch.cat([image2, labels], 0)
    last_image_dim = image1.size()[0]
    image_shape = image1.size()
    combined_pad1 = F.pad(combined1, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    combined_pad2 = F.pad(combined2, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop1 = combined_pad1[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    combined_crop2 = combined_pad2[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return (combined_crop1[:last_image_dim, :, :], combined_crop2[:last_image_dim, :, :], combined_crop1[last_image_dim:, :, :])

def random_flip(images1, images2, labels):
    
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images1 = torch.flip(images1, [1])
        images2 = torch.flip(images2, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images1 = torch.flip(images1, [2])
        images2 = torch.flip(images2, [2])
        labels = torch.flip(labels, [2])

    return images1, images2, labels


class UVGDataSet_I(data.Dataset):
    def __init__(self, root="../../data/UVG/", filelist="../../data/UVG/originalv_mm.txt", refdir='L12000'):
        with open(filelist) as f:
            folders = f.readlines()
        self.input = []
        ii = 0
        for folder in folders:
            seq = folder.rstrip()
            cnt = 96
            inputpath = []
            for i in range(cnt):
                inputpath.append(os.path.join(root, 'GT', seq, 'im'+str((i + 1)).zfill(3)+'.png'))
            self.input.append(inputpath)
            ii += 1
    
    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_images = []
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0
            h = (input_image.shape[1] // 64) * 64
            w = (input_image.shape[2] // 64) * 64
            input_images.append(input_image[:, :h, :w])
        input_images = np.array(input_images)

        return input_images

class KodakDataSet(data.Dataset):
    def __init__(self, root="../../data/Kodak24/"):
        cnt = 24
        self.inputpath = []
        for i in range(cnt):
            self.inputpath.append(os.path.join(root, 'kodim'+str((i + 1)).zfill(2)+'.png'))
    
    def __len__(self):
        return len(self.inputpath)

    def __getitem__(self, index):
        input_images = []
        filename = self.inputpath[index]
        input_image = (imageio.imread(filename).transpose(2, 0, 1)).astype(np.float32) / 255.0
        h = (input_image.shape[1] // 64) * 64
        w = (input_image.shape[2] // 64) * 64
        input_images = np.array(input_image[:, :h, :w])

        return input_images


class DataSet(data.Dataset):
    def __init__(self, latents_dtype, sigma, path="../../data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        root_path = os.path.join(os.path.dirname(path), "sequences")
        self.image_list1, self.image_list2, self.image_list3 = self.get_vimeo(rootdir=root_path ,filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width
        self.latents_dtype = latents_dtype
        self.sigma = sigma
        
        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        print("dataset find image: ", len(self.image_list1))

    def get_vimeo(self, rootdir="../../data/vimeo_septuplet/sequences/", filefolderlist="../../data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()
            
        fns_train1 = []
        fns_train2 = []
        fns_train3 = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            if int(y[-5:-4]) <= 4:
                fname1, fname2, fname3 = y[0:-5] + str(1) + '.png', y[0:-5] + str(3) + '.png', y[0:-5] + str(5) + '.png'
            elif int(y[-5:-4]) >= 6:
                fname1, fname2, fname3 = y[0:-5] + str(3) + '.png', y[0:-5] + str(5) + '.png', y[0:-5] + str(7) + '.png'
            elif int(y[-5:-4]) == 5:
                fname1, fname2, fname3 = y[0:-5] + str(2) + '.png', y[0:-5] + str(4) + '.png', y[0:-5] + str(6) + '.png'
            fns_train1 += [fname1]
            fns_train2 += [fname2]
            fns_train3 += [fname3]
        return fns_train1, fns_train2, fns_train3

    def __len__(self):
        return len(self.image_list1)

    def __getitem__(self, index):
        image1 = imageio.imread(self.image_list1[index])
        image2 = imageio.imread(self.image_list2[index])
        image3 = imageio.imread(self.image_list3[index])

        image1 = image1.astype(np.float32) / 255.0
        image2 = image2.astype(np.float32) / 255.0
        image3 = image3.astype(np.float32) / 255.0

        image1 = image1.transpose(2, 0, 1)
        image2 = image2.transpose(2, 0, 1)
        image3 = image3.transpose(2, 0, 1)
        
        image1 = torch.from_numpy(image1).float()
        image2 = torch.from_numpy(image2).float()
        image3 = torch.from_numpy(image3).float()
        
        image1, image2, image3 = random_crop_and_pad_image_and_labels(image1, image2, image3, [self.im_height, self.im_width])
        image1, image2, image3 = random_flip(image1, image2, image3)

        quant_noise_feature, quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.znoise), -0.5, 0.5)

        latents_shape = (3, self.im_height // 4, self.im_width // 4)
        latents = torch.randn(latents_shape, dtype=self.latents_dtype)
        latents = latents * self.sigma

        return image1, image2, image3, quant_noise_feature, quant_noise_z, latents
        
