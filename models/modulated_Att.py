from typing import Any

import torch
import torch.nn as nn

from torch import Tensor
from torch.autograd import Function

from compressai.layers import conv3x3, subpel_conv3x3

from models.modulator import ChannelModulator, AdaptChannelModulator 

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class MAttBlock(nn.Module):
    """Modulated Attention Block.
    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int, levels: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a_RB0 = ResidualUnit()
        self.conv_a_Modulator0 = ChannelModulator(N, levels)
        self.conv_a_RB1 = ResidualUnit()
        self.conv_a_Modulator1 = ChannelModulator(N, levels)
        self.conv_a_RB2 = ResidualUnit()

        self.conv_b_RB0 = ResidualUnit()
        self.conv_b_Modulator0 = ChannelModulator(N, levels)
        self.conv_b_RB1 = ResidualUnit()
        self.conv_b_Modulator1 = ChannelModulator(N, levels)
        self.conv_b_RB2 = ResidualUnit()
        self.conv_b_Modulator2 = ChannelModulator(N, levels)
        self.conv_b_conv0 = conv1x1(N, N)

    def forward_a(self, x: Tensor, level, alpha) -> Tensor:
        a = self.conv_a_RB0(x)
        a = self.conv_a_Modulator0(a, level, alpha)
        a = self.conv_a_RB1(a)
        a = self.conv_a_Modulator1(a, level, alpha)
        a = self.conv_a_RB2(a)
        return a

    def forward_b(self, x: Tensor, level, alpha) -> Tensor:
        b = self.conv_b_RB0(x)
        b = self.conv_b_Modulator0(b, level, alpha)
        b = self.conv_b_RB1(b)
        b = self.conv_b_Modulator1(b, level, alpha)
        b = self.conv_b_RB2(b)
        b = self.conv_b_Modulator2(b, level, alpha)
        b = self.conv_b_conv0(b)
        return b

    def forward(self, x: Tensor, level, alpha) -> Tensor:
        identity = x

        a = self.forward_a(x, level, alpha)
        b = self.forward_b(x, level, alpha)

        out = a * torch.sigmoid(b)
        out += identity
        return out


class MAttBlock_Scale(MAttBlock):
    """Modulated Attention Block with adaptive identiy modulation.
    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int, levels: int):
        super().__init__(N, levels)
        self.adapt_modulation = AdaptChannelModulator(N, levels)

    def forward(self, x: Tensor, level, alpha) -> Tensor:
        identity = self.adapt_modulation(x, level, alpha)

        a = self.forward_a(x, level, alpha)
        b = self.forward_b(x, level, alpha)

        out = a * torch.sigmoid(b)
        out += identity
        return out