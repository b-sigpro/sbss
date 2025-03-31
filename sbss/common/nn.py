# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from math import ceil

from einops import rearrange, repeat
import numpy as np

import torch  # noqa
from torch import nn
from torch.nn import functional as fn

from einops.layers.torch import Rearrange, Reduce


class ResSequential(nn.Sequential):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + super().forward(input)


class PositionalEncoding(nn.Module):
    pos_enc: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        assert d_model % 2 == 0

        self.dropout = nn.Dropout(p=dropout)

        t = torch.arange(max_len)
        tau = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pos_enc = torch.view_as_real(torch.einsum("t,c->tc", t, tau).mul(-1j).exp())
        pos_enc = rearrange(pos_enc, "t c d -> t (c d)").contiguous()
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pos_enc[: x.shape[1]])


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, n_mic: int, d_model: int):
        super().__init__()

        self.pe = nn.Parameter(torch.randn(1, n_mic, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class TACModule(nn.Module):
    def __init__(self, n_mic: int, input_dim: int):
        super().__init__()

        self.lin1 = nn.Sequential(
            Rearrange("(b m) t c -> b m t c", m=n_mic),
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
        )

        self.lin2 = nn.Sequential(
            Reduce("b m t c -> b t c", "mean"),
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
        )

        self.lin3 = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.PReLU(),
            Rearrange("b m t c -> (b m) t c", m=n_mic),
        )

    def forward(self, x: torch.Tensor):
        """implementation of TAC

        Args:
            x (torch.Tensor, shape ``[B, T, M, C]``): input hidden vectors

        Returns:
            torch.Tensor, shape ``[B, T, M, C]``: results of TAC
        """

        h0 = self.lin1(x)
        h1 = repeat(self.lin2(h0), "b t c -> b m t c", m=h0.shape[1])

        return x + self.lin3(torch.concat((h0, h1), dim=-1))


class MakeChunk(nn.Module):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__()

        self.unfold = nn.Sequential(
            nn.Unfold((chunk_size, 1), stride=(step_size, 1)),
            Rearrange("b (c t) s -> b s t c", t=chunk_size),
        )

        self.chunk_size = chunk_size
        self.step_size = step_size

    def forward(self, x: torch.Tensor):
        orig_len = x.shape[-2]
        pad_len = self.chunk_size + self.step_size * (ceil((orig_len - self.chunk_size) / self.step_size)) - orig_len

        if self.chunk_size == self.step_size:
            h = fn.pad(rearrange(x, "b t c -> b c t"), (0, pad_len))
            return rearrange(h, "b c (s t) -> b s t c", t=self.chunk_size)
        else:
            return self.unfold(fn.pad(rearrange(x, "b t c -> b c t 1"), (0, 0, 0, pad_len)))


class OverlappedAdd(nn.Module):
    def __init__(self, chunk_size: int, step_size: int):
        super().__init__()

        self.chunk_size = chunk_size
        self.step_size = step_size

    def fold(self, x: torch.Tensor, seq_len: int):
        h = rearrange(x, "b s t c -> b (c t) s")
        return fn.fold(h, (seq_len, 1), (self.chunk_size, 1), stride=(self.step_size, 1))

    def forward(self, x: torch.Tensor, orig_seq_len: int | None = None) -> torch.Tensor:
        """forward function

        Args:
            x (torch.Tensor): (B, S, T', C)
            orig_seq_len (int): sequence length

        Returns:
            torch.Tensor: (B, T, C)
        """

        if self.chunk_size == self.step_size:
            return rearrange(x, "b s t c -> b (s t) c")[:, :orig_seq_len]
        else:
            B, S, T, C = x.shape

            seq_len = self.chunk_size + self.step_size * (S - 1)

            x_ = self.fold(x, seq_len).squeeze(-1)
            ones_ = self.fold(torch.ones_like(x), seq_len).squeeze(-1)

            return rearrange(x_ / ones_, "b c t -> b t c")[:, :orig_seq_len]
