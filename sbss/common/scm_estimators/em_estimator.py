# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class EmEstimator(nn.Module):
    """Expectation-Maximization (EM) algorithm for estimating spatial covariance matrices (SCMs).

    This module iteratively estimates the SCMs using the EM algorithm
    applied to multi-channel complex-valued signals. It updates the covariance
    estimates based on posterior expectations to improve spatial separation.

    Args:
        n_iter (int): Number of EM iterations to perform.
        eps (float, optional): Small constant added to the diagonal for numerical stability.
            Defaults to 1e-6.

    Returns:
        torch.Tensor: Estimated spatial covariance tensor of shape (B, F, N, M, M),
            where B is batch size, F is number of frequency bins, N is number of sources,
            and M is number of microphones.
    """

    def __init__(self, n_iter: int, eps: float = 1e-6):
        super().__init__()

        self.n_iter = n_iter
        self.eps = eps

    def forward(self, lm: torch.Tensor, x: torch.Tensor, n_iter: int | None = None):
        B, F, M, T = x.shape
        _, _, N, _ = lm.shape

        eI = self.eps * torch.eye(M, dtype=x.dtype, device=x.device)

        H = torch.tile(torch.eye(M, device="cuda"), [B, F, N, 1, 1])
        for _ in range(self.n_iter if n_iter is None else n_iter):
            # calculate PSD
            Yk = torch.einsum("bfkt,bfkmn->bftkmn", lm, H)  # [B, F, T, K, M, M]
            Y = Yk.sum(dim=3) + eI  # [B, F, T, M, M]
            Yi = torch.linalg.inv(Y)

            # estimate image
            Yix = torch.einsum("bftmn,bfnt->bftm", Yi, x)
            YixxYi = torch.einsum("bftm,bftn->bftmn", Yix, Yix.conj())
            Z = Yk + torch.einsum("...kmn,...no,...kop->...kmp", Yk, YixxYi - Yi, Yk)

            # update H
            H = torch.einsum("bfkt,bftkmn->bfkmn", 1 / lm, Z) / T

        return H
