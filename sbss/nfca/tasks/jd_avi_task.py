# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as fn  # noqa

from einops.layers.torch import Rearrange
from torchaudio.transforms import InverseSpectrogram, Spectrogram

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


@dataclass
class Snapshot:
    xpwr: torch.Tensor
    lm: torch.Tensor
    z: torch.Tensor
    xt: torch.Tensor


class JdAviTask(OptimizerLightningModule):
    """Neural FastFCA task [Bando2023]_ implemented in PyTorch Lightning.

    Similar to :class:`sbss.nfca.tasks.AviTask`, this module optimizes the evidence lower
    bound of neural FCA, but it follows the fast variant [Bando2023]_ in which the spatial
    covariance model is factorized by an estimated joint diagonalizer ``Q`` and gain tensor
    ``g``. The encoder produces amortized posterior distributions over latent variables as 
    well as ``g``, ``Q``, and auxiliary spectrograms ``xt`` derived from the diagonalized 
    mixtures. The decoder converts latent samples into power spectral densities ``lm``; negative
    log-likelihood and KL penalties are then computed in the diagonalized domain and separation
    is performed via Wiener filtering followed by iSTFT.

    Args:
        encoder (nn.Module): Joint-diagonalization encoder returning a latent posterior
            distribution plus ``g``, ``Q``, and ``xt`` tensors described in [Bando2023]_.
        decoder (nn.Module): Maps latent variables to power spectral densities that correspond
            to ``|S|^2`` in neural FastFCA.
        n_fft (int): FFT size used during STFT preprocessing.
        hop_length (int): Hop length for the STFT/iSTFT pair.
        n_src (int): Number of sources (``N``) modeled by the task.
        beta (float): Weight applied to the KL term in the ELBO objective.
        optimizer_config (OptimizerConfig): Optimizer/lightning configuration wrapper.

    Returns:
        tuple[torch.Tensor, Snapshot]:
            - **wav_sep**: Time-domain separated waveform tensor ``[B, T]``.
            - **dump**: Snapshot with ``xpwr``, ``lm``, ``z``, and ``xt`` useful for inspection.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_fft: int,
        hop_length: int,
        n_src: int,
        beta: float,
        optimizer_config: OptimizerConfig,
    ):
        super().__init__(optimizer_config)

        self.encoder = encoder
        self.decoder = decoder

        self.stft = nn.Sequential(
            Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
            Rearrange("b m f t -> b f m t"),
        )

        # self.istft = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        self.n_src = n_src
        self.beta = beta

    def forward(self, wav: torch.Tensor, out_ch: int = 0) -> torch.Tensor:
        self.istft = InverseSpectrogram(n_fft=512, hop_length=128).cuda()
        xraw: torch.Tensor = self.stft(wav)  # [B, M, F, T]
        xpwr = xraw.abs().square().clip(1e-6)
        x = xraw / xpwr.mean(dim=(1, 2, 3), keepdim=True).sqrt()
        B, F, M, T = x.shape

        # encode
        z, g, Q, xt = self.encoder(x)

        # decode
        lm = self.decoder(z)

        # Wiener filtering
        yt = torch.einsum("bfnt,bfmn->bfmt", lm, g).add(1e-6)
        Qx_yt = torch.einsum("bfmn,bfnt->bfmt", Q, xraw) / yt
        s = torch.einsum("bfm,bfnt,bfmn,bfmt->bnft", torch.linalg.inv(Q)[..., out_ch, :], lm, g, Qx_yt)

        wav_sep = self.istft(s)

        dump = Snapshot(
            xpwr=xpwr.detach(),
            lm=lm.detach(),
            z=z.detach(),
            xt=xt.detach(),
        )

        return wav_sep, dump

    @torch.autocast("cuda", enabled=False)
    def training_step(self, wav: torch.Tensor, batch_idx, log_prefix: str = "training"):
        self.dump = None

        # stft
        x = self.stft(wav)  # [B, M, F, T]
        x /= (xpwr := x.abs().square().clip(1e-6)).mean(dim=(1, 2, 3), keepdims=True).sqrt()
        B, F, M, T = x.shape
        BFT = B * F * T

        # encode
        qz, g, Q, xt = self.encoder(x, distribution=True)
        z = qz.rsample()  # [B, D, N, T]
        _, D, *_ = z.shape

        # decode
        lm = self.decoder(z)  # [B, F, N, T]

        # calculate nll
        _, ldQ = torch.linalg.slogdet(Q)  # [B, F]
        yt = torch.einsum("bfnt,bfmn->bfmt", lm, g) + 1e-6

        nll = yt.log().sum() / BFT + torch.sum(xt.clip(1e-6) / yt) / BFT - 2 * ldQ.sum() / (B * F)

        # calculate kl
        kl = kl_divergence(qz, Normal(0, 1)).sum() / BFT

        # calculate loss
        loss = nll + self.beta * kl

        # logging
        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                f"{log_prefix}/loss": loss,
                f"{log_prefix}/nll": nll,
                f"{log_prefix}/kl": kl,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=x.shape[0],
            sync_dist=True,
        )

        self.dump = Snapshot(
            xpwr=xpwr.detach(),
            lm=lm.detach(),
            z=qz.mean.detach(),
            xt=xt.detach(),
        )

        return loss

    def validation_step(self, wav: torch.Tensor, batch_idx):
        return self.training_step(wav, batch_idx, log_prefix="validation")
