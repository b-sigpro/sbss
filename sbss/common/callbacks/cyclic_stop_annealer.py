# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from typing import Any

import torch

import lightning as lt


class CyclicStopAnnealerCallback(lt.Callback):
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
