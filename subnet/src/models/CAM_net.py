import os

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from ..entropy_models.conditional_entropy_models import EntropyBottleneck, GaussianConditional
from ..utils.stream_helper import get_downsampled_shape
from ..layers.layers import MaskedConv2d, subpel_conv3x3
from .Restormer import TransformerBlock
from ..lpips_pytorch import lpips


def save_model(model, iter, exp_name='default'):
    save_path = os.path.join("snapshot", exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join("snapshot",exp_name,f"iter{iter}.model"))


def load_model(model, f):
    print("load DCVC format")
    with open(f, 'rb') as f:
        # Load the pretrained model weights to CPU first
        pretrained_dict = torch.load(f, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Move the model to CUDA
    model = model.to(torch.device('cuda'))

    f = str(f)
    if 'iter' in f and '.model' in f:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])  # return step
    else:
        return 0


class CAM_net(nn.Module):
    def __init__(self, vae, unet, scheduler):
        '''
        More contents coming soon.
        '''
        pass
        
