# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from typing import Any

from matplotlib import pyplot as plt
import numpy as np

import torch

import lightning as lt

from sbss.nfca.tv.doa import calc_sv_point


def calc_dsbf_gain(H, mic_pos, D=360):
    F, N, T, M, _ = H.shape

    th = torch.linspace(-np.pi, np.pi, D)
    src_pos = 5.0 * torch.stack([torch.cos(th), torch.sin(th), torch.zeros(D)], dim=-1).to("cuda")  # [D, 3]

    sv = calc_sv_point(mic_pos, src_pos, F)  # [F, D, M]

    dsbf = torch.einsum("fdm,fktmn,fdn->kdt", sv.conj(), H, sv).real

    return dsbf


class VisualizerCallback(lt.Callback):
    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        dump = pl_module.dump

        B, F, N, T = dump.lm.shape
        b = np.random.choice(B)

        # plot lm and z
        logx = dump.logx[b].cpu().numpy()
        loglm = np.log(dump.lm[b].cpu().numpy())
        z = dump.z[b].cpu().numpy()

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

        # plot m, H, and u
        m = dump.m[b].cpu().numpy()

        dsbf = calc_dsbf_gain(dump.H[b], dump.mic_pos[b]).cpu().numpy()

        spk_pos = dump.spk_pos[b]
        spk_dir = spk_pos / torch.linalg.norm(spk_pos, axis=-1, keepdims=True)
        th_ora = torch.arctan2(spk_dir[..., 1], spk_dir[..., 0]).cpu().numpy() / np.pi * 180  # [N, T]
        ph_ora = torch.arcsin(spk_dir[..., 2]).cpu().numpy() / np.pi * 180  # [N, T]

        u = dump.u[b].permute(1, 0, 2)
        th_est = torch.arctan2(u[..., 1], u[..., 0]).cpu().numpy() / np.pi * 180  # [N, T]
        ph_est = torch.arcsin(u[..., 2]).cpu().numpy() / np.pi * 180  # [N, T]

        ts_ora = np.tile(np.linspace(0, T - 1, th_ora.shape[1]), [N - 1, 1])  # todo: 1 / th_ora.shape[1] 分ずれてる？
        ts_est = np.arange(T)

        gridspec_kw = dict(height_ratios=[2] + 3 * [1])
        fig, axs = plt.subplots(4, N, sharex=True, sharey="row", gridspec_kw=gridspec_kw, figsize=[4 * N, 6])
        for n, (ax1, ax2, ax3, ax4) in enumerate(axs.T):
            # imshow m
            ax1.set_title(f"Time-frequency mask m_{n}")
            ax1.imshow(m[:, n], origin="lower", aspect="auto")

            # imshow dsbf
            ax2.set_title(f"SCM H_{n}")
            ax2.imshow(dsbf[n], origin="lower", aspect="auto", extent=(0, T, -180, 180))
            ax2.scatter(ts_ora, th_ora, c="lightgray", s=5)

            # plot azimuth of u
            ax3.set_title(f"Azimuth of u_{n}")
            ax3.scatter(ts_ora, th_ora, c="lightgray", s=5)
            ax3.scatter(ts_est, th_est[n], c="red", s=5)

            # plot elevation of u
            ax4.set_title(f"Elevation of u_{n}")
            ax4.scatter(ts_ora, ph_ora, c="lightgray", s=5)
            ax4.scatter(ts_est, ph_est[n], c="red", s=5)

        axs[0, 0].set_xlim(0, T)

        axs[1, 0].set_ylim(-180, 180)
        axs[2, 0].set_ylim(-180, 180)
        axs[3, 0].set_ylim(-90, 90)

        axs[1, 0].set_yticks(np.linspace(-180, 180, 5))
        axs[2, 0].set_yticks(np.linspace(-180, 180, 5))
        axs[3, 0].set_yticks(np.linspace(-90, 90, 5))

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/tracj_mask", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
