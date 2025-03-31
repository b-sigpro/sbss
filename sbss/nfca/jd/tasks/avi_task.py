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
    xt: torch.Tensor


class AVITask(OptimizerLightningModule):
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

        self.n_src = n_src
        self.beta = beta

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

        self.dump = DumpData(
            logx=xpwr[..., 0, :].log().detach(),
            lm=lm.detach(),
            z=qz.mean.detach(),
            xt=xt.detach(),
        )

        return loss

    def validation_step(self, wav: torch.Tensor, batch_idx):
        return self.training_step(wav, batch_idx, log_prefix="validation")
