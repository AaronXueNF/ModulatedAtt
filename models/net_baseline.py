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

from compressai.models.utils import conv, deconv, update_registered_buffers, quantize_ste

from models.modulator import ChannelModulator 
from models.modulated_Att import MAttBlock
from models.entropy import CompressionModel
from models.utils_model import (
    load_pretrain_to_new_baseline, 
    load_pretrain_to_new_matt
)


class Enc_baseline(nn.Module):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__()
        self.ResidualBlockWithStride0 = ResidualBlockWithStride(3, N, stride=2)
        self.ChannelModulator0 =  ChannelModulator(N, levels)
        self.ResidualBlock0 = ResidualBlock(N, N)
        self.ChannelModulator1 =  ChannelModulator(N, levels)
        self.ResidualBlockWithStride1 = ResidualBlockWithStride(N, N, stride=2)
        self.AttentionBlock0 = AttentionBlock(N)
        self.ResidualBlock1 = ResidualBlock(N, N)
        self.ChannelModulator2 =  ChannelModulator(N, levels)
        self.ResidualBlockWithStride2 = ResidualBlockWithStride(N, N, stride=2)
        self.ChannelModulator3 =  ChannelModulator(N, levels)
        self.ResidualBlock2 = ResidualBlock(N, N)
        self.ChannelModulator4 =  ChannelModulator(N, levels)
        self.conv0 = conv3x3(N, M, stride=2)
        self.AttentionBlock1 = AttentionBlock(M)
    
    def forward(self, x, level, alpha):
        x = self.ResidualBlockWithStride0(x)
        x = self.ChannelModulator0(x, level, alpha)
        x = self.ResidualBlock0(x)
        x = self.ChannelModulator1(x, level, alpha)
        x = self.ResidualBlockWithStride1(x)
        x = self.AttentionBlock0(x)
        x = self.ResidualBlock1(x)
        x = self.ChannelModulator2(x, level, alpha)
        x = self.ResidualBlockWithStride2(x)
        x = self.ChannelModulator3(x, level, alpha)
        x = self.ResidualBlock2(x)
        x = self.ChannelModulator4(x, level, alpha)
        x = self.conv0(x)
        x = self.AttentionBlock1(x)
        return x


class Dec_baseline(nn.Module):
    def __init__(self, N, M, levels, **kwargs):
        super().__init__()
        self.AttentionBlock0 = AttentionBlock(M)
        self.ResidualBlock0 = ResidualBlock(M, M)
        self.ChannelModulator0 = ChannelModulator(M, levels)
        self.ResidualBlockUpsample0 = ResidualBlockUpsample(M, N, 2)
        self.ChannelModulator1 =ChannelModulator(N, levels)
        self.ResidualBlock1 = ResidualBlock(N, N)
        self.ChannelModulator2 = ChannelModulator(N, levels)
        self.ResidualBlockUpsample1 = ResidualBlockUpsample(N, N, 2)
        self.AttentionBlock1 = AttentionBlock(N)
        self.ResidualBlock2 = ResidualBlock(N, N)
        self.ChannelModulator3 = ChannelModulator(N, levels)
        self.ResidualBlockUpsample2 = ResidualBlockUpsample(N, N, 2)
        self.ChannelModulator4 = ChannelModulator(N, levels)
        self.ResidualBlock3 = ResidualBlock(N, N)
        self.subpel_conv0 = subpel_conv3x3(N, 3, 2)
    
    def forward(self, x, level, alpha):
        x = self.AttentionBlock0(x)
        x = self.ResidualBlock0(x)
        x = self.ChannelModulator0(x, level, alpha)
        x = self.ResidualBlockUpsample0(x)
        x = self.ChannelModulator1(x, level, alpha)
        x = self.ResidualBlock1(x)
        x = self.ChannelModulator2(x, level, alpha)
        x = self.ResidualBlockUpsample1(x)
        x = self.AttentionBlock1(x)
        x = self.ResidualBlock2(x)
        x = self.ChannelModulator3(x, level, alpha)
        x = self.ResidualBlockUpsample2(x)
        x = self.ChannelModulator4(x, level, alpha)
        x = self.ResidualBlock3(x)
        x = self.subpel_conv0(x)
        return x


MSE_LMBDA_M = [0.0016, 0.0027, 0.0045, 0.0077, 0.0130, 0.0219, 0.0371, 0.0628, 0.1063, 0.1800]
MS_SSIM_LMBDA_M = [2.11, 3.54, 5.93, 9.93, 16.64, 27.88, 46.73, 78.32, 131.26, 219.98]
MSE_LMBDA_L = [
    0.0018, 0.0029, 0.0046, 0.0074, 0.0118, 0.0189,
    0.0302, 0.0483, 0.0671, 0.0932, 0.1295, 0.1800
]
MS_SSIM_LMBDA_L = [
    2.3997, 3.8053, 6.0342, 9.5687, 15.1734, 24.0610, 
    38.1544, 60.5027, 83.5481, 115.3713, 159.3160, 219.9991
]


class cheng2020_baseline_woGMM(CompressionModel):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, metric='mse', N=192, M=192, grad_proxy='noise', **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        # define real lambda values according to distortion metric
        assert M >= N
        if metric.lower() == 'mse':
            self.lmbda = MSE_LMBDA_L if N > 192 else MSE_LMBDA_M
        elif metric.lower() == 'ms-ssim':
            self.lmbda = MS_SSIM_LMBDA_L if N > 192 else MS_SSIM_LMBDA_M
        else:
            raise NameError("Invalid distortion metric!")
        self.levels = levels = len(self.lmbda)
        
        # define modules
        self.N = int(N)
        self.M = int(M)
        self.g_a = Enc_baseline(N, M, levels)
        self.g_s = Dec_baseline(N, M, levels)

        self.h_a = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M, M, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M * 3 // 2, M * 2),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.gaussian_conditional = GaussianConditional(None)

        # training quality def
        alphas_train = [1.0, 0.5]
        levels_train = range(levels - 1)

        self.train_qualities = [(i, j) for i in levels_train for j in alphas_train ]
        self.train_qualities += [(len(self.lmbda) - 2, 0.0)]
        self.train_qualities_cycle = cycle(self.train_qualities)

        self.grad_proxy = grad_proxy.lower()


    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)


    def load_state_dict(self, state_dict, **kwargs):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, **kwargs)

    def load_state_dict_pretrain(self, pretrain_state_dict, **kwargs):
        # load state dict from pretrained model in compressai
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            pretrain_state_dict,
        )
        load_pretrain_to_new_baseline(pretrain_state_dict, self.g_a, self.g_s)
        super().load_state_dict(pretrain_state_dict, **kwargs)


    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


    def forward(self, x, level, alpha):
        y = self.g_a(x, level, alpha)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_params = self.h_s(z_hat)

        # choose use ste or add noise
        if self.grad_proxy == 'ste':
            y_hat = quantize_ste(y) if self.training else \
                self.gaussian_conditional.quantize(y, "dequantize")
        else:
            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )
 
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((hyper_params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, level, alpha)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


    def compress(self, x, level, alpha):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x, level, alpha)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)     # ????????????????????????????????????????????????
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])    # ?????????????????????

        params = self.h_s(z_hat)    # ?????????????????????

        s = 4  # scaling factor between z and y     ????????????????????????????????????????????????
        kernel_size = 5  # context prediction kernel size   ??????????????????????????????
        padding = (kernel_size - 1) // 2    # ??????padding????????????????????????????????????????????????????????????

        y_height = z_hat.size(2) * s    # ???z??????y?????????
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))  # ?????????????????????????????????padding

        y_strings = []
        for i in range(y.size(0)):  # size(0)???batch?????????????????????????????????1
            string = self._compress_ar(     # ??????????????????????????????
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}


    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()      # ??????????????????????????????cdf
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()     # ????????????cdf???????????????
        offsets = self.gaussian_conditional.offset.tolist()     # ?????????????????????????????????????????????????????????????????????pmf?????????????????????

        encoder = BufferedRansEncoder()
        symbols_list = []   # ?????????????????????
        indexes_list = []   # ??????????????????????????????

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):     # ????????????
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(      # ?????????????????????????????????????????????
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))     # ?????????????????????????????????
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)     # ??????????????????????????????????????????????????????
                scales_hat, means_hat = gaussian_params.chunk(2, 1)     # ??????????????????????????????????????????????????????????????????????????????????????????

                indexes = self.gaussian_conditional.build_indexes(scales_hat)   # ???????????????????????????????????????????????????index???????????????????????????0???

                y_crop = y_crop[:, :, padding, padding]     # ??????mask?????????????????????
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)      # ?????????????????????????????????
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat     # ???????????????????????????

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(    # ??????????????????????????????????????????index??????????????????cdf???index??????????????????cdf??????offset??????????????????
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string


    def decompress(self, strings, shape, level, alpha):
        assert isinstance(strings, list) and len(strings) == 2  # ???????????????????????????????????????list?????????2???

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)   # ???????????????
        params = self.h_s(z_hat)    # ?????????????????????

        s = 4  # scaling factor between z and y    # ?????????????????????
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s    # ?????????????????????
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(                        # ?????????0?????????????????????
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):   # ?????????????????????
            self._decompress_ar(    # ??????????????????
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))  # -padding????????????
        x_hat = self.g_s(y_hat, level, alpha).clamp_(0, 1)
        return {"x_hat": x_hat}


    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()  # ????????????
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)    # ??????????????????????????????????????????

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]  # ???????????????????????????????????????y
                ctx_p = F.conv2d(                           # ????????????
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]      # ????????????????????????????????????
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(                 # ??????????????????index???cdf?????????????????????
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv  # ????????????y?????????

