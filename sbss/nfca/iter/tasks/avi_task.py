# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as fn  # noqa

from einops.layers.torch import Rearrange
from torchaudio.transforms import Spectrogram

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


@dataclass
class DumpData:
    logx: torch.Tensor
    lm: torch.Tensor
    z: torch.Tensor


class AVITask(OptimizerLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        scm_estimator: nn.Module,
        n_fft: int,
        hop_length: int,
        n_src: int,
        beta: float,
        optimizer_config: OptimizerConfig,
    ):
        super().__init__(optimizer_config)

        self.encoder = encoder
        self.decoder = decoder
        self.scm_estimator = scm_estimator

        self.stft = nn.Sequential(
            Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
            Rearrange("b m f t -> b f m t"),
        )

        self.n_src = n_src
        self.beta = beta

    @torch.autocast("cuda", enabled=False)
    def training_step(self, wav: torch.Tensor, batch_idx, log_prefix: str = "training"):
        self.dump = None

        # initialize constants
        x = self.stft(wav)  # [B, M, F, T]
        x /= (xpwr := x.abs().square().clip(1e-6)).mean(dim=(1, 2, 3), keepdims=True).sqrt()
        B, F, M, T = x.shape
        BFT = B * F * T

        # estimate z
        qz = self.encoder(x, distribution=True)
        z = qz.rsample()  # [B, D, N, T]
        _, D, *_ = z.shape

        # calculate lm
        lm = self.decoder(z)
        lmc = lm.to(torch.complex64)

        # estimate H
        H = self.scm_estimator(lmc, x)

        # calculate nll
        eI = 1e-6 * torch.eye(M, dtype=x.dtype, device=x.device)
        Y = torch.einsum("bfkt,bfkmn->bftmn", lmc, H) + eI  # [B, F, T, M, M]
        Yi = torch.linalg.inv(Y)

        _, ldY = torch.linalg.slogdet(Y)  # [B, F, T]
        trXYi = torch.einsum("bfmt,bftmn,bfnt->bft", x.conj(), Yi, x).real  # [B, F, T]

        nll = torch.sum(ldY + trXYi) / BFT

        # calculate loss
        kl = torch.sum(kl_divergence(qz, Normal(0, 1))) / BFT

        # calculate L21 regularization term
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

        self.dump = DumpData(
            logx=xpwr[..., 0, :].log().detach(),
            lm=lm.detach(),
            z=qz.mean.detach(),
        )

        return loss

    def validation_step(self, wav: torch.Tensor, batch_idx):
        return self.training_step(wav, batch_idx, log_prefix="validation")
