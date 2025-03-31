# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from functools import partial

from einops import rearrange

import torch  # noqa
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as fn

from einops.layers.torch import Rearrange

from sbss.nfca.jd.encoders.unet_encoder import UNetBlock1d


class UNetEncoder(nn.Module):
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
