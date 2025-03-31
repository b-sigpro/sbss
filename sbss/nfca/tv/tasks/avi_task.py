# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from typing import Literal

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from einops.layers.torch import Rearrange
from torchaudio.transforms import Spectrogram

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule
from sbss.utils.distributions import FastComplexInverseWishart


@dataclass
class Snapshot:
    logx: torch.Tensor
    lm: torch.Tensor
    z: torch.Tensor
    H: torch.Tensor
    m: torch.Tensor
    G: torch.Tensor
    u: torch.Tensor
    mic_pos: torch.Tensor
    spk_pos: torch.Tensor


class AVITask(OptimizerLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_fft: int,
        hop_length: int,
        n_src: int,
        n_mic: int,
        nu: float,
        optimizer_config: OptimizerConfig,
        alpha: float = 1e-3,
        beta: float = 1.0,
        reg_mode: Literal["mode", "mean"] = "mode",
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
        self.alpha = alpha
        self.nu = nu

        if reg_mode == "mode":
            self.G_scale = nu + n_mic
        elif reg_mode == "mean":
            self.G_scale = nu - n_mic
        else:
            raise NotImplementedError()

        self.eps = 1e-8

    @torch.autocast("cuda", enabled=False)
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: torch.Tensor,
        log_prefix: str = "training",
    ):
        self.dump = None

        wav, mic_pos, spk_pos = batch

        # initialize constants
        x = self.stft(wav)  # [B, M, F, T]
        x /= (xpwr := x.abs().square().clip(1e-6)).mean(dim=(1, 2, 3), keepdims=True).sqrt()
        B, F, M, T = x.shape
        BFT = B * F * T

        # estimate z
        qz, H, m, G, u = self.encoder(x, mic_pos, distribution=True)
        z = qz.rsample()  # [B, D, N, T]
        _, D, *_ = z.shape

        # calculate lm
        lm = self.decoder(z)
        lmc = lm.to(torch.complex64)

        # estimate H
        # xx = torch.einsum("bfmt,bfnt->bftmn", x, x.conj())
        eI = 1e-6 * torch.eye(M, dtype=torch.complex64, device=self.device)

        # calculate Eq[ log p(x|lm, H) ]
        Y = torch.einsum("bfkt,bfktmn->bftmn", lmc, H) + eI  # [B, F, T, M, M]
        Yi = torch.linalg.inv(Y)

        _, ldY = torch.linalg.slogdet(Y)  # [B, F, T]
        trXYi = torch.einsum("bfmt,bftmn,bfnt->bft", x.conj(), Yi, x).real  # [B, F, T]

        nll = torch.sum(ldY + trXYi) / BFT

        # calculate KL[ q(z, th) | p(z, th) ]
        kl = torch.sum(kl_divergence(qz, Normal(0, 1))) / BFT

        # calculate log p(H|G): prior of H by G=NN(X)
        pH = FastComplexInverseWishart(self.nu, covariance_matrix=G * self.G_scale)
        reg = -pH.log_prob(H).sum() / BFT

        # calculate L21 regularization term
        loss = nll + self.beta * kl + self.alpha * reg

        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                f"{log_prefix}/loss": loss,
                f"{log_prefix}/nll": nll,
                f"{log_prefix}/kl": kl,
                f"{log_prefix}/reg": reg,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=x.shape[0],
            sync_dist=True,
        )

        self.dump = Snapshot(
            logx=xpwr[..., 0, :].log().detach(),
            lm=lm.detach(),
            z=qz.mean.detach(),
            H=H.detach(),
            m=m.detach(),
            G=G.detach(),
            u=u.detach(),
            mic_pos=mic_pos.detach(),
            spk_pos=spk_pos.detach(),
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: torch.Tensor):
        return self.training_step(batch, batch_idx, log_prefix="validation")
