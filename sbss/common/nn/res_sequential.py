# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch  # noqa
from torch import nn


class ResSequential(nn.Sequential):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + super().forward(input)
