import math
import warnings
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import SCALES_MIN, SCALES_MAX, SCALES_LEVELS, get_scale_table
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    GDN,
    MaskedConv2d,
    conv3x3,
    subpel_conv3x3,
)

# from models.entropy import GaussianMixtureConditional

from compressai.models.utils import conv, deconv, update_registered_buffers

from models.modulator import ByPass, ChannelModulator, AdaptChannelModulator
from models.modulated_Att import MAttBlock, MAttBlock_Scale
from models.entropy import CompressionModel
from models.utils_model import (
    load_pretrain_to_new_baseline, 
    load_pretrain_to_new_matt
)
from models.net_baseline import (
    cheng2020_baseline_woGMM, Enc_baseline, Dec_baseline,
    MSE_LMBDA, MS_SSIM_LMBDA
)

MSE_LMBDA_NEW = [
    0.0018, 0.0029, 0.0046, 0.0074, 0.0118, 0.0189,
    0.0302, 0.0483, 0.0671, 0.0932, 0.1295, 0.1800
]
MS_SSIM_LMBDA_NEW = [
    2.3997, 3.8053, 6.0342, 9.5687, 15.1734, 24.0610, 
    38.1544, 60.5027, 83.5481, 115.3713, 159.3160, 219.9991
]


class Enc_ModulatedAtt(Enc_baseline):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__(N=N, M=M, levels=levels, **kwargs)
        self.AttentionBlock0 = MAttBlock(N, levels)
        self.AttentionBlock1 = MAttBlock(M, levels)
    

    def forward(self, x, level, alpha):
        x = self.ResidualBlockWithStride0(x)
        x = self.ChannelModulator0(x, level, alpha)
        x = self.ResidualBlock0(x)
        x = self.ChannelModulator1(x, level, alpha)
        x = self.ResidualBlockWithStride1(x)
        x = self.AttentionBlock0(x, level, alpha)
        x = self.ResidualBlock1(x)
        x = self.ChannelModulator2(x, level, alpha)
        x = self.ResidualBlockWithStride2(x)
        x = self.ChannelModulator3(x, level, alpha)
        x = self.ResidualBlock2(x)
        x = self.ChannelModulator4(x, level, alpha)
        x = self.conv0(x)
        x = self.AttentionBlock1(x, level, alpha)
        return x


class Dec_ModulatedAtt(Dec_baseline):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__(N=N, M=M, levels=levels, **kwargs)
        self.AttentionBlock0 = MAttBlock(M, levels)
        self.AttentionBlock1 = MAttBlock(N, levels)


    def forward(self, x, level, alpha):
        x = self.AttentionBlock0(x, level, alpha)
        x = self.ResidualBlock0(x)
        x = self.ChannelModulator0(x, level, alpha)
        x = self.ResidualBlockUpsample0(x)
        x = self.ChannelModulator1(x, level, alpha)
        x = self.ResidualBlock1(x)
        x = self.ChannelModulator2(x, level, alpha)
        x = self.ResidualBlockUpsample1(x)
        x = self.AttentionBlock1(x, level, alpha)
        x = self.ResidualBlock2(x)
        x = self.ChannelModulator3(x, level, alpha)
        x = self.ResidualBlockUpsample2(x)
        x = self.ChannelModulator4(x, level, alpha)
        x = self.ResidualBlock3(x)
        x = self.subpel_conv0(x)
        return x


class cheng2020_ModulatedAtt_woGMM(cheng2020_baseline_woGMM):
    def __init__(self, metric='mse', N=256, M=320, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        # define real lambda values according to distortion metric
        if metric.lower() == 'mse':
            self.lmbda = MSE_LMBDA_NEW if N > 192 else MSE_LMBDA
        elif metric.lower() == 'ms-ssim':
            self.lmbda = MS_SSIM_LMBDA_NEW if N > 192 else MS_SSIM_LMBDA
        else:
            raise NameError("Invalid distortion metric!")
        self.levels = levels = len(self.lmbda)

        # training quality def
        alphas_train = [1.0, 0.5]
        levels_train = range(levels - 1)

        self.train_qualities = [(i, j) for i in levels_train for j in alphas_train ]
        self.train_qualities += [(len(self.lmbda) - 2, 0.0)]
        self.train_qualities_cycle = cycle(self.train_qualities)

        self.g_a = Enc_ModulatedAtt(N, M, levels)
        self.g_s = Dec_ModulatedAtt(N, M, levels)


    def load_state_dict_pretrain(self, pretrain_state_dict, **kwargs):
        # load state dict from pretrained model in compressai
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            pretrain_state_dict,
        )
        load_pretrain_to_new_matt(pretrain_state_dict, self.g_a, self.g_s)
        super().load_state_dict(pretrain_state_dict, **kwargs)



class Enc_AdaptMAtt(Enc_ModulatedAtt):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__(N=N, M=M, levels=levels, **kwargs)
        self.ChannelModulator0 = AdaptChannelModulator(N, levels)
        self.ChannelModulator1 = AdaptChannelModulator(N, levels)
        self.ChannelModulator2 = AdaptChannelModulator(N, levels)
        self.ChannelModulator3 = AdaptChannelModulator(N, levels)
        self.ChannelModulator4 = AdaptChannelModulator(N, levels)
    

class Dec_AdaptMAtt(Dec_ModulatedAtt):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__(N=N, M=M, levels=levels, **kwargs)
        self.ChannelModulator0 = AdaptChannelModulator(M, levels)
        self.ChannelModulator1 = AdaptChannelModulator(N, levels)
        self.ChannelModulator2 = AdaptChannelModulator(N, levels)
        self.ChannelModulator3 = AdaptChannelModulator(N, levels)
        self.ChannelModulator4 = AdaptChannelModulator(N, levels)
    

class cheng2020_AdaptMAtt_woGMM(cheng2020_baseline_woGMM):
    def __init__(self, metric='mse', N=256, M=320, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        # define real lambda values according to distortion metric
        if metric.lower() == 'mse':
            self.lmbda = MSE_LMBDA_NEW
        elif metric.lower() == 'ms-ssim':
            self.lmbda = MS_SSIM_LMBDA_NEW
        else:
            raise NameError("Invalid distortion metric!")
        self.levels = levels = len(self.lmbda)

        # training quality def
        alphas_train = [1.0, 0.5]
        levels_train = range(levels - 1)

        self.train_qualities = [(i, j) for i in levels_train for j in alphas_train ]
        self.train_qualities += [(len(self.lmbda) - 2, 0.0)]
        self.train_qualities_cycle = cycle(self.train_qualities)

        self.g_a = Enc_AdaptMAtt(N, M, levels)
        self.g_s = Dec_AdaptMAtt(N, M, levels)


    def load_state_dict_pretrain(self, pretrain_state_dict, **kwargs):
        # load state dict from pretrained model in compressai
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            pretrain_state_dict,
        )
        load_pretrain_to_new_matt(pretrain_state_dict, self.g_a, self.g_s)
        super().load_state_dict(pretrain_state_dict, **kwargs)


class Enc_AdaptScaleMAtt(Enc_ModulatedAtt):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__(N=N, M=M, levels=levels, **kwargs)
        self.AttentionBlock0 = MAttBlock_Scale(N, levels)
        self.AttentionBlock1 = MAttBlock_Scale(M, levels)

        self.ChannelModulator0 = AdaptChannelModulator(N, levels)
        self.ChannelModulator1 = ByPass(N, levels)
        self.ChannelModulator2 = ByPass(N, levels)
        self.ChannelModulator3 = AdaptChannelModulator(N, levels)
        self.ChannelModulator4 = ChannelModulator(N, levels)
    

class Dec_AdaptScaleMAtt(Dec_ModulatedAtt):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__(N=N, M=M, levels=levels, **kwargs)
        self.AttentionBlock0 = MAttBlock_Scale(M, levels)
        self.AttentionBlock1 = MAttBlock_Scale(N, levels)

        self.ChannelModulator0 = ChannelModulator(M, levels)
        self.ChannelModulator1 = AdaptChannelModulator(N, levels)
        self.ChannelModulator2 = ByPass(N, levels)
        self.ChannelModulator3 = ByPass(N, levels)
        self.ChannelModulator4 = AdaptChannelModulator(N, levels)
    

class cheng2020_AdaptScaleMAtt_woGMM(cheng2020_baseline_woGMM):
    def __init__(self, metric='mse', N=256, M=320, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        # define real lambda values according to distortion metric
        if metric.lower() == 'mse':
            self.lmbda = MSE_LMBDA_NEW
        elif metric.lower() == 'ms-ssim':
            self.lmbda = MS_SSIM_LMBDA_NEW
        else:
            raise NameError("Invalid distortion metric!")
        self.levels = levels = len(self.lmbda)

        # training quality def
        alphas_train = [1.0, 0.5]
        levels_train = range(levels - 1)

        self.train_qualities = [(i, j) for i in levels_train for j in alphas_train ]
        self.train_qualities += [(len(self.lmbda) - 2, 0.0)]
        self.train_qualities_cycle = cycle(self.train_qualities)

        self.g_a = Enc_AdaptScaleMAtt(N, M, levels)
        self.g_s = Dec_AdaptScaleMAtt(N, M, levels)


    def load_state_dict_pretrain(self, pretrain_state_dict, **kwargs):
        # load state dict from pretrained model in compressai
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            pretrain_state_dict,
        )
        load_pretrain_to_new_matt(pretrain_state_dict, self.g_a, self.g_s)
        super().load_state_dict(pretrain_state_dict, **kwargs)