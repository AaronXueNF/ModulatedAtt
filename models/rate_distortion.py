import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, metric='mse'):
        super().__init__()
        assert metric.lower() in {'mse', 'ms-ssim'}
        self.metric = metric.lower()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        if self.metric == 'ms-ssim':
            out["D_loss"] = 1 - ms_ssim(output["x_hat"], target, data_range=1.0, size_average=True)
        else:
            out["D_loss"] = (255 ** 2) * self.mse(output["x_hat"], target)

        out["R_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["loss"] = lmbda * out["D_loss"] + out["R_loss"]

        return out