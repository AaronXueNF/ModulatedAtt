import torch
import torch.nn as nn
import torch.nn.functional as F


class ByPass(nn.Module):
    def __init__(self, channels, levels=10):
        super(ByPass, self).__init__()

    def forward(self, x, level, alpha):
        return x


class ChannelModulator(nn.Module):
    """
    InterpCA module for latent variable
    """

    def __init__(self, channels, levels=10):
        super(ChannelModulator, self).__init__()
        self.levels = levels
        self.channels = channels

        self.scalar = nn.Parameter(torch.ones(size=[levels, channels]), requires_grad=True)
        self.softplus = nn.Softplus()

    def forward(self, x, level, alpha):
        # restrict level and alpha in reasonable range.
        alpha = torch.clamp(torch.tensor(alpha), 0.0, 1.0).unsqueeze(-1).to(x.device)
        level = torch.clamp(torch.tensor(level), 0, self.levels - 2).to(x.device)

        # alpha = 0, Interpolated = level+1; alpha = 1, Interpolated = level
        scalar0 = self.scalar[level]
        scalar1 = self.scalar[level + 1]
        scalar_final = alpha * scalar0 + (1 - alpha) * scalar1
        scalar_final = self.softplus(scalar_final)
        scalar_final = scalar_final.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [M]  -->  [1,M,1,1]

        x = x * scalar_final
        return x


class AdaptChannelModulator(nn.Module):
    """
    InterpCA module with energy based adapt
    """

    def __init__(self, channels, levels=10, adj_range=0.5):
        super(AdaptChannelModulator, self).__init__()
        self.levels = levels
        self.channels = channels

        # learnable scalar
        self.scalar = nn.Parameter(torch.ones(size=[levels, channels]), requires_grad=True)
        self.softplus = nn.Softplus()

        # adaptive conv
        self.adj_range = adj_range
        self.adj_add = 1 - adj_range / 2
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x, level, alpha):
        # restrict level and alpha in reasonable range.
        alpha = torch.clamp(torch.tensor(alpha), 0.0, 1.0).unsqueeze(-1).to(x.device)
        level = torch.clamp(torch.tensor(level), 0, self.levels - 2).to(x.device)

        # alpha = 0, Interpolated = level+1; alpha = 1, Interpolated = level
        scalar0 = self.scalar[level]
        scalar1 = self.scalar[level + 1]
        scalar_final = alpha * scalar0 + (1 - alpha) * scalar1
        scalar_final = self.softplus(scalar_final)
        scalar_final = scalar_final.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [M]  -->  [1,M,1,1]

        # energy based adaptive calculation
        x_adapt = x * x     # energy
        x_adapt = self.avgpool(x_adapt)     # TODO: WARNING, may unstable if x is too large!
        x_adapt = self.conv0(x_adapt)
        x_adapt = self.softplus(x_adapt)
        x_adapt = self.conv1(x_adapt)
        scalar_adapt = self.sigmoid(x_adapt) * self.adj_range + self.adj_add

        scalar_final = scalar_final * scalar_adapt
        x = x * scalar_final
        return x