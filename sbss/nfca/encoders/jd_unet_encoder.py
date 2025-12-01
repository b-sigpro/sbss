# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from functools import partial

from einops import rearrange

import torch  # noqa
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as fn

from einops.layers.torch import Rearrange

from sbss.nfca.encoders.unet_encoder import Conv1dBlock


class UNetBlock1d(nn.Module):
    def __init__(
        self,
        n_stft: int,
        n_mic: int,
        io_ch: int,
        xt_ch: int,
        mid_ch: int,
        ksize: int,
        n_layers: int,
        use_xt: bool = False,
        use_r: bool = True,
    ):
        super().__init__()

        self.cnv0 = nn.Sequential(Conv1dBlock(io_ch + xt_ch if use_xt else io_ch, mid_ch, 1))
        self.cnvs = nn.ModuleList(
            [Conv1dBlock(mid_ch, mid_ch, ksize, 1 if ll == 0 else 2, mid_ch) for ll in range(n_layers)]
        )
        self.nrmact = nn.Sequential(
            nn.GroupNorm(1, mid_ch),
            nn.PReLU(mid_ch),
        )

        self.cnv_h = nn.Sequential(nn.Conv1d(mid_ch, io_ch, 1))

        if use_r:
            self.cnv_r = nn.Sequential(
                nn.Conv1d(mid_ch, n_mic * n_stft, 1),
                nn.Sigmoid(),
                Rearrange("b (f m) t -> b f m t", m=n_mic),
            )
        else:
            self.cnv_r = lambda h: None

    def forward(self, input, h_xt=None):
        """
        Parameters
        ----------
        input : (B, C1, T) Tensor
        xt : (B, C2, M, T) Tensor
        """

        # in_ch -> mid_ch
        h = input if h_xt is None else torch.concat([input, h_xt], dim=1)
        h = self.cnv0(h)

        # downsample
        hs = []
        for cnv in self.cnvs:
            h = cnv(h)
            hs.append(h)

        # upsample
        for h_ in hs[::-1][1:]:
            h = fn.interpolate(h, scale_factor=2)[..., : h_.shape[2]] + h_

        # mid_ch -> in_ch
        h = self.nrmact(h)
        res_output = self.cnv_h(h)

        r = self.cnv_r(h)

        return res_output + input, r


class JdUNetEncoder(nn.Module):
    """U-Net-based encoder for multichannel audio representation learning.

    This encoder extracts latent variables and spatial gain masks from
    multichannel spectrograms using a hierarchical convolutional structure.
    It combines frequency-phase features with iterative diagonalization
    through a series of UNet blocks, producing both latent distributions
    and source-wise gain estimates.

    Args:
        n_fft (int): FFT size used for spectrogram computation.
        n_mic (int): Number of input microphone channels.
        n_src (int): Number of output sources to separate.
        dim_latent (int): Dimensionality of the latent variable space.
        diagonalizer (nn.Module): Module used to perform covariance diagonalization.
        io_ch (int, optional): Number of intermediate feature channels. Defaults to 256.
        xt_ch (int, optional): Number of channels for intermediate spatial features. Defaults to 512.
        mid_ch (int, optional): Number of channels in intermediate UNet blocks. Defaults to 512.
        n_blocks (int, optional): Number of stacked UNet blocks. Defaults to 8.
        n_layers (int, optional): Number of convolutional layers per UNet block. Defaults to 5.
        ksize (int, optional): Kernel size of 1D convolutions. Defaults to 5.

    Returns:
        tuple:
            qz (torch.distributions.Normal or torch.Tensor): Latent variable distribution or its mean.
            g (torch.Tensor): Source gain mask tensor of shape (B, F, M, N).
            Q (torch.Tensor): Estimated diagonalizer matrix of shape (B, F, M, M).
            xt (torch.Tensor): Spatially transformed spectrogram features.
    """

    def __init__(
        self,
        n_fft: int,
        n_mic: int,
        n_src: int,
        dim_latent: int,
        diagonalizer: nn.Module,
        io_ch: int = 256,
        xt_ch: int = 512,
        mid_ch: int = 512,
        n_blocks: int = 8,
        n_layers: int = 5,
        ksize: int = 5,
    ):
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.bn0 = nn.BatchNorm1d(n_stft)
        self.cnv0 = nn.Conv1d((2 * n_mic - 1) * n_stft, io_ch, 1)

        UNetBlock1d_ = partial(UNetBlock1d, n_stft, n_mic, io_ch, xt_ch, mid_ch, ksize, n_layers)
        self.cnv_list = nn.ModuleList(
            [UNetBlock1d_(False, True)]
            + [UNetBlock1d_(True, True) for _ in range(n_blocks - 1)]
            + [UNetBlock1d_(True, False)]
        )

        self.diagonalizer = diagonalizer

        self.cnv_xt = nn.Sequential(
            Rearrange("b f m t -> b (f m) t", f=n_stft, m=n_mic),
            nn.Conv1d(n_mic * n_stft, xt_ch, 1, bias=False),
        )

        self.cnv_z = nn.Sequential(
            nn.Conv1d(io_ch, 2 * dim_latent * n_src, 1),
            Rearrange("b (c d n) t -> c b d n t", c=2, d=dim_latent, n=n_src),
        )

        self.cnv_g = nn.Sequential(
            nn.Conv1d(io_ch, n_stft * n_mic * n_src, 1),
            nn.Sigmoid(),
            Rearrange("b (f m n) t -> b f m n t", f=n_stft, m=n_mic, n=n_src),
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

        # pre-convolution
        h = self.cnv0(h)

        xt, Q = None, torch.tile(torch.eye(M, dtype=torch.complex64, device="cuda"), [B, F, 1, 1])  # [B, F, K, M, M]
        h, r = self.cnv_list[0](h)
        for cnv in self.cnv_list[1:]:  # type: ignore
            Q, xt = self.diagonalizer(r, Q, x)

            h_xt = self.cnv_xt(xt.clip(1e-6).log())
            h, r = cnv(h, h_xt)

        z_mu, z_sig_ = self.cnv_z(h)  # [B, 2, D, N, T]
        qz = Normal(z_mu, fn.softplus(z_sig_) + 1e-6) if distribution else z_mu

        g: torch.Tensor = torch.einsum("bfmnt,bfmt->bfmn", self.cnv_g(h), xt)  # type: ignore
        g = g / g.mean(dim=2, keepdim=True).clip(1e-6)

        return qz, g, Q, xt
