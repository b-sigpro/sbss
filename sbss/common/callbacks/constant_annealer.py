# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import torch

import lightning as lt


class ConstantAnnealerCallback(lt.Callback):
    """Lightning callback that temporarily overrides a module attribute during training.

    This callback sets a specified attribute (e.g., a learning rate or coefficient)
    in the Lightning module to a constant value for a given training period, and
    then restores its original value afterward.

    Args:
        name (str): Name of the attribute in the Lightning module to override.
        value (float): Constant value to assign to the attribute during the annealing period.
        period (float): Fraction of the total training steps during which the attribute
            is held constant (e.g., 0.5 means for the first half of training).

    Returns:
        None
    """

    def __init__(self, name: str, value: float, period: float):
        self.name = name

        self.value = value
        self.period = period

        self.initialized = False
        self.orig_value = None

    def on_train_batch_start(self, trainer: lt.Trainer, pl_module: Any, batch: torch.Tensor, batch_idx: int):
        if not self.initialized:
            self.orig_value = getattr(pl_module, self.name)

        step = trainer.global_step / trainer.num_training_batches

        if step < self.period:
            setattr(pl_module, self.name, self.value)
        else:
            setattr(pl_module, self.name, self.orig_value)
