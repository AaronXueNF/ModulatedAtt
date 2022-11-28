import math
import warnings

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


def _load_data(Module, pretrain_state_dict, name, idx):
    for k, v in Module.state_dict().items():
        v.copy_(pretrain_state_dict[f'{name}.{idx}.' + k])


def load_pretrain_to_new_baseline(pretrain_state_dict, g_a, g_s):
    name = 'g_a'
    _load_data(g_a.ResidualBlockWithStride0, pretrain_state_dict, name, 0)
    _load_data(g_a.ResidualBlock0, pretrain_state_dict, name, 1)
    _load_data(g_a.ResidualBlockWithStride1, pretrain_state_dict, name, 2)
    _load_data(g_a.AttentionBlock0, pretrain_state_dict, name, 3)
    _load_data(g_a.ResidualBlock1, pretrain_state_dict, name, 4)
    _load_data(g_a.ResidualBlockWithStride2, pretrain_state_dict, name, 5)
    _load_data(g_a.ResidualBlock2, pretrain_state_dict, name, 6)
    _load_data(g_a.conv0, pretrain_state_dict, name, 7)
    _load_data(g_a.AttentionBlock1, pretrain_state_dict, name, 8)

    name = 'g_s'
    _load_data(g_s.AttentionBlock0, pretrain_state_dict, name, 0)
    _load_data(g_s.ResidualBlock0, pretrain_state_dict, name, 1)
    _load_data(g_s.ResidualBlockUpsample0, pretrain_state_dict, name, 2)
    _load_data(g_s.ResidualBlock1, pretrain_state_dict, name, 3)
    _load_data(g_s.ResidualBlockUpsample1, pretrain_state_dict, name, 4)
    _load_data(g_s.AttentionBlock1, pretrain_state_dict, name, 5)
    _load_data(g_s.ResidualBlock2, pretrain_state_dict, name, 6)
    _load_data(g_s.ResidualBlockUpsample2, pretrain_state_dict, name, 7)
    _load_data(g_s.ResidualBlock3, pretrain_state_dict, name, 8)
    _load_data(g_s.subpel_conv0, pretrain_state_dict, name, 9)


def _load_data_matt(Module, pretrain_state_dict, name, idx):
    skip_a, skip_b = 0, 0
    for k, v in Module.state_dict().items():
        temp = k.split('.')
        skip_a += 1 if (temp[0] == 'conv_a' and temp[2] == 'scalar') else 0
        skip_b += 1 if (temp[0] == 'conv_b' and temp[2] == 'scalar') else 0
        if 'scalar' in k:
            continue
        temp[1] = str(int(temp[1]) - (skip_a if temp[0] == 'conv_a' else skip_b))
        target_k = '.'.join(temp)
        v.copy_(pretrain_state_dict[f'{name}.{idx}.' + target_k])


def load_pretrain_to_new_matt(pretrain_state_dict, g_a, g_s):
    name = 'g_a'
    _load_data(g_a.ResidualBlockWithStride0, pretrain_state_dict, name, 0)
    _load_data(g_a.ResidualBlock0, pretrain_state_dict, name, 1)
    _load_data(g_a.ResidualBlockWithStride1, pretrain_state_dict, name, 2)
    _load_data_matt(g_a.AttentionBlock0, pretrain_state_dict, name, 3)
    _load_data(g_a.ResidualBlock1, pretrain_state_dict, name, 4)
    _load_data(g_a.ResidualBlockWithStride2, pretrain_state_dict, name, 5)
    _load_data(g_a.ResidualBlock2, pretrain_state_dict, name, 6)
    _load_data(g_a.conv0, pretrain_state_dict, name, 7)
    _load_data_matt(g_a.AttentionBlock1, pretrain_state_dict, name, 8)

    name = 'g_s'
    _load_data_matt(g_s.AttentionBlock0, pretrain_state_dict, name, 0)
    _load_data(g_s.ResidualBlock0, pretrain_state_dict, name, 1)
    _load_data(g_s.ResidualBlockUpsample0, pretrain_state_dict, name, 2)
    _load_data(g_s.ResidualBlock1, pretrain_state_dict, name, 3)
    _load_data(g_s.ResidualBlockUpsample1, pretrain_state_dict, name, 4)
    _load_data_matt(g_s.AttentionBlock1, pretrain_state_dict, name, 5)
    _load_data(g_s.ResidualBlock2, pretrain_state_dict, name, 6)
    _load_data(g_s.ResidualBlockUpsample2, pretrain_state_dict, name, 7)
    _load_data(g_s.ResidualBlock3, pretrain_state_dict, name, 8)
    _load_data(g_s.subpel_conv0, pretrain_state_dict, name, 9)

