# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from einops import rearrange

import torch  # noqa
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as fn

from einops.layers.torch import Rearrange


class DepSepConv1d(nn.Sequential):
    def __init__(self, io_ch, mid_ch, ksize, dilation):
        super().__init__(
            nn.Conv1d(io_ch, mid_ch, 1),
            nn.PReLU(mid_ch),
            nn.GroupNorm(mid_ch, mid_ch),
            #
            nn.Conv1d(mid_ch, mid_ch, ksize, padding=(ksize - 1) // 2 * dilation, dilation=dilation, groups=mid_ch),
            nn.PReLU(mid_ch),
            nn.GroupNorm(mid_ch, mid_ch),
            #
            nn.Conv1d(mid_ch, io_ch, 1),
        )

    def forward(self, x):
        return x + super().forward(x)


class DilConvBlock1d(nn.Sequential):
    def __init__(self, io_ch, mid_ch, ksize, n_layers):
        super().__init__(*[DepSepConv1d(io_ch, mid_ch, ksize, 2**ll) for ll in range(n_layers)])


class DilcnvEncoder(nn.Module):
    """Encoder module using dilated depthwise separable convolutions for multichannel spectrograms.

    This encoder extracts feature representations from multichannel complex spectrograms.
    It first applies batch normalization and logarithmic power compression to the reference
    channel, then concatenates normalized phase differences between channels. The combined
    features are processed by a stack of dilated convolutional blocks to produce latent
    variables, which can optionally be interpreted as Gaussian distributions.

    Args:
        n_fft (int): FFT size used to determine the number of frequency bins.
        n_mic (int): Number of microphone channels in the input.
        n_src (int): Number of sources to estimate in the latent space.
        dim_latent (int): Dimensionality of the latent variable.
        io_ch (int, optional): Number of input/output channels for convolution blocks.
            Defaults to 256.
        mid_ch (int, optional): Number of intermediate channels in convolution blocks.
            Defaults to 512.
        n_blocks (int, optional): Number of stacked convolutional blocks. Defaults to 4.
        n_layers (int, optional): Number of dilated convolution layers per block. Defaults to 8.
        ksize (int, optional): Kernel size for dilated convolutions. Defaults to 3.

    Returns:
        torch.distributions.Normal | torch.Tensor:
            If ``distribution=True``, returns a Normal distribution with
            mean and scale tensors shaped ``[B, D, N, T]``.
            Otherwise, returns the mean tensor directly.
    """

    def __init__(
        self,
        n_fft: int,
        n_mic: int,
        n_src: int,
        dim_latent: int,
        io_ch: int = 256,
        mid_ch: int = 512,
        n_blocks: int = 4,
        n_layers: int = 8,
        ksize: int = 3,
    ):
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.bn0 = nn.BatchNorm1d(n_stft)
        self.cnv = nn.Sequential(
            nn.Conv1d((2 * n_mic - 1) * n_stft, io_ch, 1),
            *[DilConvBlock1d(io_ch, mid_ch, ksize, n_layers) for _ in range(n_blocks)],
        )

        self.cnv_z = nn.Sequential(
            nn.Conv1d(io_ch, 2 * dim_latent * n_src, 1),
            Rearrange("b (c d n) t -> c b d n t", c=2, d=dim_latent, n=n_src),
        )

    def forward(self, x: torch.Tensor, distribution: bool = False):
        """
        Parameters
        ----------
        x : (B, F, M, T) Tensor
            Multichannel spectrogram
        distribution : bool, optional
            If True, the returns will be distributions. Defaults is False.
        """

        B, F, M, T = x.shape

        # generate feature vectors
        logx = self.bn0(x[..., 0, :].abs().square().clip(1e-6).log())  # [B, F, T]

        ph = x[..., 1:, :] / (x[..., 0, None, :] + 1e-6)  # [B, F, M-1, T]
        ph /= torch.abs(ph).clip(1e-6)
        ph = rearrange(torch.view_as_real(ph), "b f m t c -> b (f m c) t")

        h = torch.concat([logx, ph], dim=1)  # [B, C, T]

        # main convolutions
        h = self.cnv(h)

        z_mu, z_sig_ = self.cnv_z(h)  # [B, 2, D, N, T]
        qz = Normal(z_mu, fn.softplus(z_sig_) + 1e-6) if distribution else z_mu

        return qz
