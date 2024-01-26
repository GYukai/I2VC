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

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))

def load_model(model, f):
    print("load DCVC format")
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])    #return step
    else:
        return 0

class CAM_net(nn.Module):
    def __init__(self, vae, unet, scheduler):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        out_channel_N = 192
        out_channel_M = 192
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M
        self.bitEstimator_z = BitEstimator(out_channel_N)

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.gaussian_encoder = GaussianEncoder()

        self.contextualEncoder = contextualEncoder()
        self.contextualDecoder_part1 = contextualDecoder_part1()

        self.contextualDecoder_part_2 = nn.Sequential(
            nn.Conv2d(out_channel_N+3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            *[TransformerBlock(dim=out_channel_N, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(4)],
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.priorEncoder = PriorEncoder_net(out_channel_M, out_channel_N)
        self.priorDecoder = PriorDecoder_net(in_channel=out_channel_N, out_channel=out_channel_M)

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 9 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(
            out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1
        )
        self.gaussian_conditional = GaussianConditional(None)
        

        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False

    def quantize(self, inputs, mode, means=None):
        assert(mode == "dequantize")
        outputs = inputs.clone()
        outputs -= means
        outputs = torch.round(outputs)
        outputs += means
        return outputs

    def feature_probs_based_sigma(self, feature, mean, sigma):
        outputs = feature
        values = outputs - mean
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    def iclr18_estrate_bits_z(self, z):
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def update(self, force=False):
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    def forward(self, input_image, latents, lmd = None, lmd_boundary = None, previous_frame = None, feature_frame=None, quant_noise_feature=None, quant_noise_z=None):
        extra_kwargs = {}
        extra_kwargs["eta"] = 1
        # feature = self.feature_extract(input_image)
        # context = self.context_refine(feature)
# TODO 1 in
        feature_vae = self.vae.encode(input_image).latents
        feature = self.contextualEncoder(feature_vae, lmd, lmd_boundary)
        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        
        params = self.priorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            compressed_y_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_y_renorm = torch.round(feature_renorm)

        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        low_res_latents = self.contextualDecoder_part1(compressed_y_renorm, lmd, lmd_boundary)
        # TODO out
        for t in self.scheduler.timesteps:
            latents_input = torch.cat([latents, low_res_latents], dim=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            noise_pred = self.unet(latents_input, t).sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample
        recon_image = self.vae.decode(latents).sample

        clipped_recon_image = recon_image.clamp(0., 1.)

        total_bits_y, _ = self.feature_probs_based_sigma(compressed_y_renorm, means_hat, scales_hat)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp = bpp_y + bpp_z 

        # distortion
        distortion = torch.mean((recon_image - input_image).pow(2))
        lps_distortion = lpips(recon_image, input_image, net_type='squeeze')
        
        return clipped_recon_image, distortion, lps_distortion, bpp 


# =============================================================================================
# =============================================================================================
# =============================================================================================
out_channel_N = 192
out_channel_M = 192
middle_channel = 3

class PriorEncoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorEncoder_net, self).__init__()
        self.l1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class PriorDecoder_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorDecoder_net, self).__init__()
        self.l1 = nn.ConvTranspose2d(in_channel, out_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (in_channel + in_channel))))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.ConvTranspose2d(out_channel, out_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (out_channel + out_channel))))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

class Entropy_parameters_net(nn.Module):
    def __init__(self, channel):
        super(Entropy_parameters_net, self).__init__()
        self.l1 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l1.bias.data, 0.01)
        self.r1 = nn.LeakyReLU(inplace=True)
        self.l2 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l2.bias.data, 0.01)
        self.r2 = nn.LeakyReLU(inplace=True)
        self.l3 = nn.Conv2d(channel, channel, 1)
        torch.nn.init.xavier_normal_(self.l3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.l3.bias.data, 0.01)
    def forward(self, x):
        x = self.r1(self.l1(x))
        x = self.r2(self.l2(x))
        x = self.l3(x)
        return x

# =============================================================================================
# =============================================================================================
# =============================================================================================
    

class spatial_gating_unit(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv_1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)       
        self.conv_2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)        
        self.conv_3 = nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.relu_1(x_1)
        
        x_2 = self.conv_2(x)
        x_2 = self.relu_2(x_2)
        x_2 = self.conv_3(x_2)
        i_mask = self.sigmoid(x_2)
        
        x_gated = x_1 * i_mask + x
        
        return x_gated, i_mask
        
class sf_mlp(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.linear_1 = nn.Linear(2, 50)        # zyf
        self.relu = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(50, num_filters)
        
    def forward(self, lmd):
        cond = self.linear_1(lmd)
        cond = self.relu(cond)
        cond = self.linear_2(cond)

        w = cond.exp()
        return w
        
        
class spatial_scaling_network(nn.Module):
    def __init__(self, num_filters):
        super().__init__()        
        self.sf_mlp = sf_mlp(num_filters)
        
    def forward(self, i_mask, lmd_normed):
        B,C,H,W = i_mask.shape
        mask_map = torch.mean(i_mask,axis=1).view(B,1,H,W)  # (B,1,H,W)
        cond_map = torch.ones_like(mask_map)    # (B,1,H,W)
        # print(cond_map.shape,lmd_normed.shape)
        cond_map = lmd_normed * cond_map 
        sf_input = torch.cat([mask_map, cond_map], axis=1)  # (B,2,H,W)
        sf_input = sf_input.permute(0,3,2,1)    # (B,W,H,2)
        scaling_factor = self.sf_mlp(sf_input)  # (B,W,H,N)
        scaling_factor = scaling_factor.permute(0,3,2,1)    # (B,N,H,W)
        
        return scaling_factor

class contextualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        self.gdn1 = GDN(out_channel_N)
        self.res1=ResBlock_LeakyReLU_0_Point_1(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1)
        self.gdn2 = GDN(out_channel_N)
        self.res2 = ResBlock_LeakyReLU_0_Point_1(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        
        
        self.spatial_gating_unit_1 = spatial_gating_unit(out_channel_N)
        self.spatial_gating_unit_2 = spatial_gating_unit(out_channel_N)
        self.spatial_gating_unit_3 = spatial_gating_unit(out_channel_N)
        self.spatial_scaling_network_1 = spatial_scaling_network(num_filters=out_channel_N)
        self.spatial_scaling_network_2 = spatial_scaling_network(num_filters=out_channel_N)
        self.spatial_scaling_network_3 = spatial_scaling_network(num_filters=out_channel_N)

    def forward(self, x, lmd, lmd_boundary):
        lmd_normal = (lmd/lmd_boundary)

        x = self.gdn1(self.conv1(x))
        x, i_mask = self.spatial_gating_unit_1(x)
        sf = self.spatial_scaling_network_1(i_mask, lmd_normal)
        x = x*sf
        x = self.res1(x)
        x = self.gdn2(self.conv2(x))
        x, i_mask = self.spatial_gating_unit_2(x)
        sf = self.spatial_scaling_network_2(i_mask, lmd_normal)
        x = x*sf
        x = self.res2(x)
        x = self.gdn3(self.conv3(x))
        x, i_mask = self.spatial_gating_unit_3(x)
        sf = self.spatial_scaling_network_3(i_mask, lmd_normal)
        x = x*sf
        return self.conv4(x)

class contextualDecoder_part1(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = subpel_conv3x3(out_channel_M, out_channel_N, 2)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = subpel_conv3x3(out_channel_N, middle_channel, 2)

        self.spatial_gating_unit_1 = spatial_gating_unit(out_channel_N)
        self.spatial_scaling_network_1 = spatial_scaling_network(num_filters=out_channel_N)

    def forward(self, x, lmd, lmd_boundary):
        lmd_normal = (lmd/lmd_boundary)
        x = self.deconv1(x)
        x, i_mask = self.spatial_gating_unit_1(x)
        sf = self.spatial_scaling_network_1(i_mask, lmd_normal)
        x = self.igdn1(x*sf)
        return self.deconv2(x)
