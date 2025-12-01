#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import os
from pathlib import Path

from omegaconf import DictConfig

import numpy as np

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config
from rich.progress import track

import soundfile as sf


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("split", type=str, default="train")
    args = parser.parse_args()

    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    split_path = dataset_path / args.split
    dst_path = split_path / "mix"
    dst_path.mkdir(exist_ok=True)

    src_filename_list = list((split_path / "mix-clean").glob("*.wav"))
    src_filename_list.sort()

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        src_filename_list = src_filename_list[start:end]

    for src_filename in track(src_filename_list):
        wav, sr = sf.read(src_filename)

        wav = wav[:, : config.n_mics]
        wav += np.sqrt(np.mean(wav**2)) * 10 ** (-config.snr / 20) * np.random.randn(*wav.shape)

        assert np.all(np.abs(wav) < 1)

        sf.write(dst_path / src_filename.name, wav, sr, "PCM_24")


if __name__ == "__main__":
    main()
