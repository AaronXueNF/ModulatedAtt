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


MSE_LMBDA = [0.0016, 0.0027, 0.0045, 0.0077, 0.0130, 0.0219, 0.0371, 0.0628, 0.1063, 0.1800]
MS_SSIM_LMBDA = [2.11, 3.54, 5.93, 9.93, 16.64, 27.88, 46.73, 78.32, 131.26, 219.98]


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
        if metric.lower() == 'mse':
            self.lmbda = MSE_LMBDA
        elif metric.lower() == 'ms-ssim':
            self.lmbda = MS_SSIM_LMBDA
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

        z_strings = self.entropy_bottleneck.compress(z)     # 完全分解的熵模型，压缩超先验信息
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])    # 重建超先验信息

        params = self.h_s(z_hat)    # 超先验解码信息

        s = 4  # scaling factor between z and y     需要根据超先验模型计算，填到这里
        kernel_size = 5  # context prediction kernel size   上下文模型卷积核尺寸
        padding = (kernel_size - 1) // 2    # 手动padding，上下文模型的串行计算需要在后续手动实现

        y_height = z_hat.size(2) * s    # 由z反推y的尺寸
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))  # 这个是重建的图像，加入padding

        y_strings = []
        for i in range(y.size(0)):  # size(0)是batch维度，只压缩一张图像为1
            string = self._compress_ar(     # 每次循环压缩一张图像
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
        cdf = self.gaussian_conditional.quantized_cdf.tolist()      # 获取高斯分布的量化后cdf
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()     # 获取每个cdf的有效长度
        offsets = self.gaussian_conditional.offset.tolist()     # 获取高斯模型的每个偏差，这个偏差为每个高斯模型pmf分布的中心位置

        encoder = BufferedRansEncoder()
        symbols_list = []   # 存储量化后符号
        indexes_list = []   # 存储符号对应分布序号

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):     # 遍历图像
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(      # 卷积得到当前位置上下文预测结果
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))     # 拼接超先验与上下文信息
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)     # 去除最后两个无用的维度，得到高斯参数
                scales_hat, means_hat = gaussian_params.chunk(2, 1)     # 高斯参数分为两半，前一半为尺度（方差）信息，后一半为均值信息

                indexes = self.gaussian_conditional.build_indexes(scales_hat)   # 根据尺度，计算每个隐变量对应的分布index（这个分布均值认为0）

                y_crop = y_crop[:, :, padding, padding]     # 获取mask卷积中心的像素
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)      # 量化为符号，并减去均值
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat     # 加上均值，得到重建

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(    # 调用熵编码器，根据符号列表、index列表、不同的cdf（index指示采用哪个cdf）、offset进行熵编码。
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string


    def decompress(self, strings, shape, level, alpha):
        assert isinstance(strings, list) and len(strings) == 2  # 确保输入包含先验、超先验（list长度为2）

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)   # 解压超先验
        params = self.h_s(z_hat)    # 计算超先验参数

        s = 4  # scaling factor between z and y    # 这部分同编码器
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s    # 这部分同编码器
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(                        # 构建全0的重建先验信息
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):   # 对于每一张图像
            self._decompress_ar(    # 关键解码函数
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))  # -padding实现裁剪
        x_hat = self.g_s(y_hat, level, alpha).clamp_(0, 1)
        return {"x_hat": x_hat}


    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()  # 同编码器
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)    # 重要，将字符序列存入熵解码器

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]  # 循环获取上下文卷积核尺寸的y
                ctx_p = F.conv2d(                           # 手动卷积
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]      # 获取对应位置的超先验信息
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(                 # 输入高斯分布index、cdf相关信息，解码
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv  # 将重建的y填回去

