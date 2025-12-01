# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

import torch

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class WavHDF5Dataset(torch.utils.data.Dataset):
    """Dataset class for loading and preprocessing multi-channel audio data stored in HDF5 format.

    This dataset wraps an ``HDF5Dataset`` using a caching layer for faster access. It supports
    random microphone permutation for data augmentation and optional random cropping
    of the waveform based on the specified duration and sampling rate.

    Args:
        dataset_path (Path | str): Path to the HDF5 dataset directory or file.
        duration (int | None, optional): Target duration of audio clips in seconds. If None,
            the full audio is used.
        sr (int | None, optional): Sampling rate used to calculate the crop length when
            ``duration`` is specified.
        randperm_mic (bool, optional): Whether to randomly permute microphone channels.
            Defaults to True.
        grp_list (Path | str | list[str] | None, optional): List or path specifying group names
            to load from the HDF5 dataset. If None, all groups are used.

    Returns:
        torch.utils.data.Dataset: A PyTorch dataset yielding preprocessed multi-channel waveforms.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        duration: int | None = None,
        sr: int | None = None,
        randperm_mic: bool = True,
        grp_list: Path | str | list[str] | None = None,
    ) -> None:
        super().__init__()

        self._dataset = CachedDataset(HDF5Dataset(dataset_path, grp_list))

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
