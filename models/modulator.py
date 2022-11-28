import torch
import torch.nn as nn
import torch.nn.functional as F


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