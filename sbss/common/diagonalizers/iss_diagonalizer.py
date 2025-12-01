# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class ISSDiagonalizer(nn.Module):
    """Iterative Source Steering (ISS) algorithm for diagonalizing spatial covariance matrices.

    This module performs iterative updates of the demixing matrix ``Q`` based on the
    input mixture ``x`` and inverse power spectrum density ``r``. It can be used as
    part of spatial source separation algorithms such as IVA and FastMNMF.

    Args:
        n_iter (int, optional): Number of ISS iterations to perform. Defaults to 1.
        eps (float, optional): Regularization constant added to covariance diagonals
            to prevent numerical instability. Defaults to 1e-6.
        eps2 (float, optional): Small value used to clip denominator terms during
            normalization to avoid division by zero. Defaults to 1e-6.
        eps3 (float, optional): Minimum clipping value for the inverse  power estimates in
            ``r`` to stabilize computations. Defaults to 1e-3.
        norm_q (bool, optional): Whether to normalize the demixing matrix ``Q`` at
            the end of the iteration. Defaults to False.

    Returns:
        nn.Module: A PyTorch module that outputs the updated demixing matrix ``Q`` and
        the corresponding source power estimates ``xt`` after applying the ISS updates.
    """

    def __init__(
        self, n_iter: int = 1, eps: float = 1e-6, eps2: float = 1e-6, eps3: float = 1e-3, norm_q: bool = False
    ):
        super().__init__()

        self.n_iter = n_iter

        self.eps = eps
        self.eps2 = eps2
        self.eps3 = eps3

        self.norm_q = norm_q

    def forward(self, r, Q, x):
        """
        Parameters
        ----------
        r : (B, F, M, T) Tensor
        Q : (B, F, M, M) Tensor
        x : (B, F, M, T) Tensor
        """

        _, _, M, T = x.shape

        V = torch.einsum("...kt,...mt,...nt->...kmn", r.clip(self.eps3), x, x.conj()) / T
        trV = torch.einsum("...mm->...", V)[..., None, None].real
        V = V + trV.maximum(torch.ones_like(trV)).to(V.dtype) * self.eps * torch.eye(M, device="cuda")

        for _ in range(self.n_iter):
            for k in range(M):
                q = Q[..., k, :]
                Vq = torch.einsum("...kmn,...n->...km", V, q.conj())

                qVq = torch.einsum("...m,...km->...k", q, Vq).real.clip(self.eps2)
                v = torch.einsum("...km,...km->...k", Q, Vq) / qVq.to(x.dtype)
                v[..., k] = 1 - qVq[..., k] ** -0.5

                Q = Q - torch.einsum("...m,...n->...mn", v, q)

        Qx = Q @ x
        xt = Qx.real**2 + Qx.imag**2  # torch.abs(Qx) ** 2

        if self.norm_q:
            scale = xt.mean(dim=(1, 2, 3), keepdim=True)
            xt = xt / scale
            Q = Q / scale.clip(1e-6).sqrt().to(x.dtype)

        return Q, xt
