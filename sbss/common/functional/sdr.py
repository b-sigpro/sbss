# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch


def signal_to_noise_ratio(
    wav_est: torch.Tensor,
    wav_tgt: torch.Tensor,
    max_sdr: float | None = None,
    zero_mean: bool = False,
    eps: float = 1e-8,
):
    tau = 0.0 if max_sdr is None else 10 ** (-max_sdr / 10)

    if zero_mean:
        wav_est = wav_est - wav_est.mean(dim=-1, keepdim=True)
        wav_tgt = wav_tgt - wav_tgt.mean(dim=-1, keepdim=True)

    e_pwr = (wav_est - wav_tgt).square().sum(dim=-1)
    s_pwr = wav_tgt.square().sum(dim=-1)

    thresh = tau * s_pwr
    snr = 10 * (torch.log10(s_pwr + eps) - torch.log10(e_pwr + thresh.clamp_min(eps)))

    return snr


def scale_invariant_signal_to_distortion_ratio(
    wav_est: torch.Tensor,
    wav_tgt: torch.Tensor,
    max_sdr: float | None = None,
    zero_mean: bool = False,
    eps: float = 1e-8,
):
    if zero_mean:
        wav_est = wav_est - wav_est.mean(dim=-1, keepdim=True)
        wav_tgt = wav_tgt - wav_tgt.mean(dim=-1, keepdim=True)

    a = (wav_tgt * wav_est).sum(dim=-1, keepdim=True) / wav_tgt.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    wav_tgt = a * wav_tgt

    return signal_to_noise_ratio(
        wav_est=wav_est,
        wav_tgt=wav_tgt,
        max_sdr=max_sdr,
        zero_mean=False,
        eps=eps,
    )
