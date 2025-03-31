# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from pathlib import Path

import numpy as np

import torch

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class WavHDF5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: Path | str,
        duration: int | None = None,
        sr: int | None = None,
        randperm_mic: bool = True,
    ) -> None:
        super().__init__()

        self._dataset = CachedDataset(HDF5Dataset(dataset_path))

        self.duration = duration
        self.sr = sr

        self.randperm_mic = randperm_mic

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        item = self._dataset[index]
        wav = item["wav"]

        if self.duration is not None:
            duration = self.sr * self.duration

            t_start = np.random.randint(0, wav.shape[1] - duration + 1)
            t_end = t_start + duration

            wav = wav[:, t_start:t_end]

        if self.randperm_mic:
            wav = wav[torch.randperm(wav.shape[0])]

        return wav
