# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch
from torch import nn


class EMEstimator(nn.Module):
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
