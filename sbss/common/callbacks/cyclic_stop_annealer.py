# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import torch

import lightning as lt


class CyclicStopAnnealerCallback(lt.Callback):
    """Lightning callback that cyclically increases a module attribute and stops after a period.

    This callback linearly increases a specified attribute (such as a learning rate or coefficient)
    within each cycle until a defined period is reached, after which the attribute is reset to its
    original value and no longer updated.

    Args:
        name (str): Name of the attribute in the Lightning module to modify.
        cycle (int): Number of training epochs or fraction of total steps that define one cycle.
        max_value (float): Maximum value reached during the cyclic increase.
        period (int, optional): Duration of cyclic behavior before stopping and restoring the original value.
            Defaults to 0.

    Returns:
        None
    """

    def __init__(self, name: str, cycle: int, max_value: float, period: int = 0):
        self.name = name
        self.cycle = cycle
        self.max_value = max_value

        self.period = period

        self.initialized = False
        self.orig_value = None

    def on_train_batch_start(self, trainer: lt.Trainer, pl_module: Any, batch: torch.Tensor, batch_idx: int):
        if not self.initialized:
            self.orig_value = getattr(pl_module, self.name)

        step = trainer.global_step / trainer.num_training_batches

        if step < self.period:
            setattr(pl_module, self.name, self.max_value * min(2 * (step % self.cycle) / self.cycle, 1))
        else:
            setattr(pl_module, self.name, self.orig_value)
