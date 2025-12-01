# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch  # noqa
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()

        self.num_channels = num_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(num_channels))
        self.bias = nn.Parameter(torch.Tensor(num_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        mu = torch.mean(x, dim=(1, 3), keepdim=True)
        sig = torch.sqrt(torch.mean((x - mu) ** 2, dim=(1, 3), keepdim=True) + self.eps)

        return (x - mu) / sig * self.weight[:, None, None] + self.bias[:, None, None]


class ConvBlock2d(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(nn.Conv2d(in_ch, out_ch, 1), nn.PReLU(), LayerNorm(out_ch))


class ResConvBlock2d(ConvBlock2d):
    def __init__(self, io_ch):
        super().__init__(io_ch, io_ch)

    def forward(self, x):
        return x + super().forward(x)


class ResDecoder(nn.Module):
    """Decoder module that transforms latent representations into magnitude spectrogram estimates.

    This module applies several 2D convolutional and residual blocks to reconstruct
    a positive-valued spectrogram-like output from the input latent tensor.
    The final output is passed through a Softplus activation and offset slightly
    to avoid numerical instability.

    Args:
        n_fft (int): FFT size used in the spectrogram, determining the number of frequency bins.
        dim_latent (int): Dimension of the latent representation.
        io_ch (int, optional): Number of intermediate convolution channels. Defaults to 256.
        n_layers (int, optional): Number of convolutional layers (including residual ones). Defaults to 3.

    Returns:
        torch.Tensor: Estimated magnitude spectrogram tensor of shape [B, F, N, T].
    """

    def __init__(self, n_fft: int, dim_latent: int, io_ch: int = 256, n_layers: int = 3):
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.cnv = nn.Sequential(
            nn.Conv2d(dim_latent, io_ch, 1),
            *[ResConvBlock2d(io_ch) for ll in range(n_layers - 1)],
            nn.Conv2d(io_ch, n_stft, 1),
            nn.Softplus(),
        )

    def forward(self, z):
        """
        Parameters
        ----------
        z : [B, D, N, T]
        """

        return self.cnv(z) + 1e-6  # [B, F, N, T]
