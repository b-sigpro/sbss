# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import numpy as np

import torch


def fibonacci_sphere(n_samples=1000):
    ph = np.pi * (3.0 - np.sqrt(5.0))
    ii = np.arange(n_samples)

    y = 1 - 2 * (ii / (n_samples - 1))
    r = np.sqrt(1 - y * y)

    th = ph * ii
    x = np.cos(th) * r
    z = np.sin(th) * r

    return np.stack([x, y, z], axis=-1)


def calc_sv_point(mic_pos: torch.Tensor, spk_pos: torch.Tensor, F: int, sr: int = 16000, c: float = 340):
    """
    Parameters
    ----------
    mic_pos : (..., M, 3) Tensor
    spk_pos : (..., N, 3) Tensor

    Returns
    ----------
    sv : (..., F, N, M) Tensor

    """

    # if mic_pos.dim() != spk_pos.dim():
    #     raise ValueError()

    tau = torch.linalg.norm(mic_pos.unsqueeze(-3) - spk_pos.unsqueeze(-2), dim=-1) / c * sr  # [..., N, M]
    tau = tau - tau[..., 0][..., None]

    f = torch.linspace(0, 1, F, device=mic_pos.device).reshape([F, 1, 1])
    sv = torch.exp(-1j * torch.pi * f * tau.unsqueeze(-3))  # [..., F, N, M]

    return sv


def calc_dsbf_doa(x, mic_pos, fib_pos, src_scale=5, sr=16000, c=340):
    """
    Parameters
    ----------
    X : (B, F, M, T) Tensor
    mic_pos : (B, M, 3) Tensor
    fib_pos : (D, 3) Tensor
    sr: sampling rate
    c: speed of sound

    tdoa: (B, F, 3, T)
    """

    B, F, M, T = x.shape

    sv = calc_sv_point(mic_pos, src_scale * fib_pos, F)

    dsbf_indices = torch.einsum("bfdm,bfmt->bftd", sv.conj(), x).abs().argmax(dim=-1)  # [B, F, T]
    doa = torch.stack([torch.take(fib_pos[:, d], dsbf_indices) for d in range(3)], dim=-1)

    return doa


def calc_araki_doa(mic_pos: torch.Tensor, x: torch.Tensor, sr: int = 16000, c: float = 340):
    """
    Parameters
    ----------
    mic_pos : (B, M, 3) Tensor
    X : (B, F, M, T) Tensor
        Multichannel spectrogram
    sr: sampling rate
    c: speed of sound

    tdoa: (B, F, 3, T)
    """

    B, F, M, T = x.shape

    D = mic_pos[:, :, None] - mic_pos[:, None, :]
    D = D.reshape(B, M**2, 3)
    D_pinv = torch.linalg.pinv(D)  # D_pinv: [B, M**2, 3]

    q = torch.einsum("bfmt,bfnt->bfmnt", x, x.conj())  # [B, F, M, M, T]
    q = q.reshape([B, F, M**2, T])  # q: [B, F ,M**2, T]
    q = torch.atan2(q.imag, q.real)

    f = torch.linspace(0, sr // 2, F, device=mic_pos.device, dtype=torch.float32).reshape([1, F, 1, 1])
    q /= 2 * torch.pi * f
    q[:, 0] = torch.zeros(q[:, 0].shape, device=mic_pos.device)

    doa = c * torch.einsum("bdm, bfmt->bfdt", D_pinv, q).real  # [B, F, 3, T]
    doa /= torch.linalg.norm(doa, dim=-2, keepdims=True).clip(1e-6)

    return doa
