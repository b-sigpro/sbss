# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import torch

import lightning as lt


class CyclicAnnealerCallback(lt.Callback):
    """Lightning callback that cyclically varies a module attribute during training.

    This callback modifies a specified attribute (e.g., a learning rate or coefficient)
    in the Lightning module according to a cyclic schedule. The value increases linearly
    within each cycle and resets afterward. An optional initial period can use a different
    maximum value before switching to the main cycle.

    Args:
        name (str): Name of the attribute in the Lightning module to modify.
        cycle (int): Number of training epochs or fraction of total steps that define one cycle.
        max_value (float): Maximum value reached during the cycle.
        ini_period (int, optional): Initial period before cyclic behavior starts. Defaults to 0.
        ini_max_value (float, optional): Maximum value used during the initial period. Defaults to 1.0.

    Returns:
        None
    """

    def __init__(self, name: str, cycle: int, max_value: float, ini_period: int = 0, ini_max_value: float = 1.0):
        self.name = name
        self.cycle = cycle
        self.max_value = max_value

        self.ini_period = ini_period
        self.ini_max_value = ini_max_value

    def on_train_batch_start(self, trainer: lt.Trainer, pl_module: Any, batch: torch.Tensor, batch_idx: int):
        step = trainer.global_step / trainer.num_training_batches

        max_value = self.ini_max_value if step < self.ini_period else self.max_value
        setattr(pl_module, self.name, max_value * min(2 * (step % self.cycle) / self.cycle, 1))
