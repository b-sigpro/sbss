# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WarmUp(LambdaLR):
    def __init__(self, optimizer: Optimizer, n_steps: int = 1000):
        super().__init__(optimizer, lambda epoch: min((epoch - 1) / (n_steps - 1), 1.0))
