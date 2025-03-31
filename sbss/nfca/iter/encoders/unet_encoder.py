# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from einops import rearrange

import torch  # noqa
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as fn

from einops.layers.torch import Rearrange


class Conv1dBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, ksize: int, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv1d(in_ch, out_ch, ksize, stride=stride, padding=(ksize - 1) // 2, groups=groups, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.PReLU(out_ch),
        )


class UNetBlock1d(nn.Module):
    def __init__(
        self,
        io_ch: int,
        mid_ch: int,
        ksize: int,
        n_layers: int,
    ):
        super().__init__()

        self.cnv_enc = Conv1dBlock(io_ch, mid_ch, 1)
        self.cnvs = nn.ModuleList(
            [Conv1dBlock(mid_ch, mid_ch, ksize, 1 if ll == 0 else 2, mid_ch) for ll in range(n_layers)]
        )

        self.cnv_dec = nn.Sequential(nn.GroupNorm(1, mid_ch), nn.PReLU(mid_ch), nn.Conv1d(mid_ch, io_ch, 1))

    def forward(self, x):
        # in_ch -> mid_ch
        h = self.cnv_enc(x)

        # downsample
        hs = []
        for cnv in self.cnvs:
            h = cnv(h)
            hs.append(h)

        # upsample
        for h_ in hs[::-1][1:]:
            h = fn.interpolate(h, scale_factor=2)[..., : h_.shape[2]] + h_

        # mid_ch -> in_ch
        h = self.cnv_dec(h)

        return h + x


class UNetEncoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        n_mic: int,
        n_src: int,
        dim_latent: int,
        io_ch: int = 256,
        mid_ch: int = 512,
        n_blocks: int = 8,
        n_layers: int = 5,
        ksize: int = 5,
    ):
        super().__init__()

        n_stft = n_fft // 2 + 1

        self.bn0 = nn.BatchNorm1d(n_stft)
        self.cnv = nn.Sequential(
            nn.Conv1d((2 * n_mic - 1) * n_stft, io_ch, 1),
            *[UNetBlock1d(io_ch, mid_ch, ksize, n_layers) for _ in range(n_blocks)],
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
