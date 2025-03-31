# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from typing import Any

import torch

import lightning as lt


class ConstantAnnealerCallback(lt.Callback):
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
