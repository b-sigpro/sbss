# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch  # noqa
from torch import nn

from einops.layers.torch import Rearrange


class ResConvBlock2d(nn.Sequential):
    def __init__(self, io_ch):
        super().__init__(
            nn.LayerNorm(io_ch),
            nn.Linear(io_ch, io_ch),
            nn.PReLU(),
            nn.Linear(io_ch, io_ch),
        )

    def forward(self, x):
        return x + super().forward(x)


class ResLinearDecoder(nn.Module):
    """Decoder module that reconstructs a magnitude spectrogram from a latent representation.

    This module applies a series of linear and residual layers to convert the input
    latent feature tensor into a positive-valued spectrogram-like output. It supports
    optional zero-masking of noise-related latent dimensions before decoding.

    Args:
        n_fft (int): FFT size used in the spectrogram, determining the number of frequency bins.
        dim_latent (int): Dimension of the input latent representation.
        io_ch (int, optional): Number of hidden channels in the intermediate layers. Defaults to 256.
        n_layers (int, optional): Number of linear layers including residual ones. Defaults to 3.
        dim_latent_noi (int | None, optional): Starting index of noise-related latent dimensions to zero out.
            If None, no masking is applied. Defaults to None.
        n_noi (int | None, optional): Number of noise-related channels to mask.
            Defaults to 1 if ``dim_latent_noi`` is given.

    Returns:
        torch.Tensor: Estimated magnitude spectrogram tensor of shape [B, F, N, T].
    """

    def __init__(
        self,
        n_fft: int,
        dim_latent: int,
        io_ch: int = 256,
        n_layers: int = 3,
        dim_latent_noi: int | None = None,
        n_noi: int | None = None,
    ):
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.cnv = nn.Sequential(
            Rearrange("b d n t -> b n t d"),
            nn.Linear(dim_latent, io_ch),
            *[ResConvBlock2d(io_ch) for ll in range(n_layers - 1)],
            nn.Linear(io_ch, n_stft),
            nn.Softplus(),
            Rearrange("b n t f -> b f n t"),
        )

        self.dim_latent_noi = dim_latent_noi
        self.n_noi = 1 if n_noi is None else n_noi

    def forward(self, z):
        """
        Parameters
        ----------
        z : [B, D, N, T]
        """

        if self.dim_latent_noi is not None:
            z[:, self.dim_latent_noi :, -self.n_noi :, :] = 0.0

        return self.cnv(z) + 1e-6  # [B, F, N, T]
