# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import torch

import lightning as lt


class PsdVisualizerCallback(lt.Callback):
    """Callback to visualize intermediate tensors during model validation.

    This callback plots the observation, predicted power spectral densities (PSDs),
    and latent variables stored in ``pl_module.dump`` at the beginning and end
    of each validation phase. The resulting figures are logged to the experiment
    logger (e.g., TensorBoard).

    Args:
        trainer (lightning.Trainer): PyTorch Lightning trainer handling the training loop.
        pl_module (Any): The Lightning module that must contain a ``dump`` attribute with
            fields ``pwrx``, ``lm``, and ``z`` representing observation, PSD, and latent tensors.
        tag (str, optional): Label to identify whether the visualization is from training
            or validation. Defaults to ``"training"``.

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
        F, N, T = dump.lm.shape

        logx = dump.xpwr[:, 0, :].log10().mul_(10)
        loglm = dump.lm.log10().mul_(10)

        # plot observation and PSDs
        gridspec_kw = dict(height_ratios=[2] + N * [2, 1])
        fig, axs = plt.subplots(1 + (2 * N), 1, sharex=True, gridspec_kw=gridspec_kw, figsize=[8, 2 + 3 * N])

        axs[0].imshow(logx, origin="lower", aspect="auto")

        lmmin, lmmax = loglm.min(), loglm.max()
        zmin, zmax = dump.z.min(), dump.z.max()
        for n, (ax1, ax2) in enumerate(axs[1 : 1 + 2 * N].reshape(-1, 2)):
            ax1.imshow(loglm[..., n, :], origin="lower", aspect="auto", vmin=lmmin, vmax=lmmax)

            ax2.plot(dump.z[..., n, :].T)
            ax2.set_xlim(0, T - 1)
            ax2.set_ylim(zmin, zmax)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{prefix}/dump", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
