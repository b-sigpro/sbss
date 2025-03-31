# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import lightning as lt


class VisualizerCallback(lt.Callback):
    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        dump = pl_module.dump

        # numpyize dumped variables
        B, F, N, T = dump.lm.shape
        b = np.random.choice(B)

        logx = dump.logx[b].cpu().numpy()
        loglm = dump.lm[b].log().cpu().numpy()
        z = dump.z[b].cpu().numpy()

        # plot observation and PSDs
        gridspec_kw = dict(height_ratios=[2] + N * [2, 1])
        fig, axs = plt.subplots(1 + (2 * N), 1, sharex=True, gridspec_kw=gridspec_kw, figsize=[8, 2 + 3 * N])

        axs[0].imshow(logx, origin="lower", aspect="auto")

        lmmin, lmmax = loglm.min(), loglm.max()
        zmin, zmax = z.min(), z.max()
        for n, (ax1, ax2) in enumerate(axs[1 : 1 + 2 * N].reshape(-1, 2)):
            ax1.imshow(loglm[..., n, :], origin="lower", aspect="auto", vmin=lmmin, vmax=lmmax)

            ax2.plot(z[..., n, :].T)
            ax2.set_xlim(0, T - 1)
            ax2.set_ylim(zmin, zmax)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/dump", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
