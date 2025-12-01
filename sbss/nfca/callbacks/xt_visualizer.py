# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import torch

import lightning as lt


class XtVisualizerCallback(lt.Callback):
    """Callback to visualize the time-frequency energy (Xt) of the model output during validation.

    This callback generates spectrogram-like plots from the `dump.xt` tensor and logs
    them to TensorBoard at the start and end of validation. It assumes that the Lightning module
    has a `dump` attribute containing `xt` (the time-frequency representation) and `data_name`.

    Args:
        trainer (lightning.Trainer): PyTorch Lightning trainer handling the training/validation loop.
        pl_module (Any): Lightning module being trained or validated.
        tag (str, optional): Label prefix for TensorBoard logging. Defaults to `"training"`.

    Returns:
        None
    """

    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        dump = pl_module.dump

        prefix = f"{tag}"
        if hasattr(dump, "data_name"):
            prefix += f"/{dump.data_name}"

        # select a random batch element
        B, *_ = dump.xpwr.shape
        b = np.random.choice(B)

        dump = dump.__class__(**{k: v[b].cpu() if isinstance(v, torch.Tensor) else v for k, v in vars(dump).items()})

        logxt = dump.xt.log10().mul_(10).cpu().numpy()
        _, M, _ = logxt.shape

        # plot xt
        fig, axs = plt.subplots(M, 1, sharex=True, figsize=[8, 1.5 * M])
        vmax = logxt.max()
        vmin = vmax - 80
        for m, ax in enumerate(axs):
            ax.imshow(logxt[..., m, :], origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{prefix}/xt", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
