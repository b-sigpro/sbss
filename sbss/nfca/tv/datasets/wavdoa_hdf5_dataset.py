# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from __future__ import annotations

from pathlib import Path

import numpy as np

import torch

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class WavDoaHDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        duration: int | None = None,
        rotaug: bool = True,
    ) -> None:
        super().__init__()

        self._dataset = CachedDataset(HDF5Dataset(dataset_path))

        self.duration = duration
        self.rotaug = rotaug

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        item = self._dataset[index]

        wav = item["wav"]
        mic_pos = item["mic_pos"]
        spk_pos = item["spk_pos"]

        if self.rotaug:
            th = np.random.uniform(0, np.pi)
            rot = torch.tensor(
                [[np.cos(+th), np.sin(-th), 0], [np.sin(+th), np.cos(+th), 0], [0, 0, 1]], dtype=torch.float32
            )

            mic_pos = torch.einsum("ij,mj->mi", rot, mic_pos)
            spk_pos = torch.einsum("ij,kmj->kmi", rot, spk_pos)

        return wav, mic_pos, spk_pos
