# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch  # noqa
from torch import nn
from torch.nn import functional as fn
from torch.distributions import Normal

from einops import rearrange
from einops.layers.torch import Rearrange

from sbss.nfca.tv.nn import ResSequential
from sbss.nfca.tv.doa import fibonacci_sphere, calc_sv_point, calc_dsbf_doa
from sbss.nfca.tv.window import calc_exponential_window


class DepSepConv1d(ResSequential):
    def __init__(self, io_ch, mid_ch, ksize, dilation, dropout):
        super().__init__(
            nn.Conv1d(io_ch, mid_ch, 1),
            nn.PReLU(),
            nn.GroupNorm(mid_ch, mid_ch),
            nn.Dropout(dropout),
            #
            nn.Conv1d(mid_ch, mid_ch, ksize, padding=(ksize - 1) // 2 * dilation, dilation=dilation, groups=mid_ch),
            nn.PReLU(),
            nn.GroupNorm(mid_ch, mid_ch),
            nn.Dropout(dropout),
            #
            nn.Conv1d(mid_ch, io_ch, 1),
            nn.Dropout(dropout),
        )


class DilConvBlock1d(nn.Sequential):
    def __init__(self, io_ch, mid_ch, ksize, n_layers, dropout):
        super().__init__(*[DepSepConv1d(io_ch, mid_ch, ksize, 2**ll, dropout) for ll in range(n_layers)])


class DilatedConvEncoder(nn.Module):
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
        dropout: float = 0.1,
        decay_H: float = 0.01,
        decay_G: float = 0.01,
        dlt_H: float = 1e-6,
        dlt_G: float = 1e-3,
        gamma: float = 1e-1,
        sv_scale: int = 5,
    ) -> None:
        super().__init__()

        n_stft = n_fft // 2 + 1

        # initialize hyperparameters
        self.register_buffer("dI_H", dlt_H * torch.eye(n_mic))
        self.register_buffer("dI_G", dlt_G * torch.eye(n_mic))
        self.eps = 1e-8

        self.decay_H = decay_H
        self.decay_G = decay_G

        self.gamma = gamma

        self.sv_scale = sv_scale

        self.register_buffer("fib_pos", torch.as_tensor(fibonacci_sphere(1000), dtype=torch.float32))

        # initialize nn modules
        self.bn0 = nn.BatchNorm1d(n_stft)

        self.cnv = nn.Sequential(
            nn.Conv1d(4 * n_stft, io_ch, 1),
            *[DilConvBlock1d(io_ch, mid_ch, ksize, n_layers, dropout) for _ in range(n_blocks)],
        )

        self.cnv_z = nn.Sequential(
            nn.Conv1d(io_ch, 2 * dim_latent * n_src, 1),
            Rearrange("b (c d n) t -> c b d n t", c=2, d=dim_latent, n=n_src),
        )
        self.cnv_u = nn.Sequential(
            nn.Conv1d(io_ch, n_src * 3, 1),
            Rearrange("b (n c) t -> b t n c", n=n_src, c=3),
        )
        self.cnv_m = nn.Sequential(
            nn.Conv1d(io_ch, n_src * n_stft, 1),
            Rearrange("b (f n) t -> b f n t", f=n_stft, n=n_src),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mic_pos: torch.Tensor, distribution: bool = False):
        B, F, M, T = x.shape

        logx = self.bn0(torch.log(torch.abs(x[..., 0, :]) ** 2 + self.eps))  # [B, F, T]
        doa = rearrange(calc_dsbf_doa(x, mic_pos, self.fib_pos), "b f t c -> b (f c) t")  # [B, F, T, 3]

        h = torch.cat([logx, doa], dim=1)  # [B, C, T]

        # main convolutions
        h = self.cnv(h)

        # calculate z
        z_mu, z_sig_ = self.cnv_z(h)  # [B, D, N, T]
        z = Normal(z_mu, fn.softplus(z_sig_) + 1e-6) if distribution else z_mu

        # calculate G
        u_ = fn.normalize(self.cnv_u(h), dim=-1)  # [B, T, N, 3]

        win_u = calc_exponential_window(T, self.decay_G)
        u = fn.normalize(torch.einsum("ts,bsnp->btnp", win_u, u_), dim=-1)

        sv = calc_sv_point(mic_pos[:, None], self.sv_scale * u, F)  # [B, T, F, N, M]
        G = torch.einsum("btfkm,btfkn->bfktmn", sv, sv.conj()) + self.dI_G  # [B, F, N, T, M, M]
        G[:, :, -1] = torch.eye(M, device=x.device)

        # calculate H
        m = self.cnv_m(h)  # [B, F, N, T]
        mc = m.to(torch.complex64)

        xnrm = x / torch.linalg.norm(x, dim=2, keepdims=True).clip(1e-6)

        win_H = calc_exponential_window(T, self.decay_H).to(torch.complex64)
        win_Hc = win_H.to(torch.complex64)

        H = torch.einsum("ts,bfks,bfms,bfns->bfktmn", win_Hc, mc, xnrm, xnrm.conj())
        H[:, :, -1] = torch.einsum("bft,bfmt,bfnt->bfmn", mc[:, :, -1], xnrm, xnrm.conj()).unsqueeze(2)
        H = H + self.gamma * G + self.dI_H
        H = H / (torch.einsum("bfktmm->bfkt", H).real / M + 1e-6)[..., None, None]

        return z, H, m, G, u
