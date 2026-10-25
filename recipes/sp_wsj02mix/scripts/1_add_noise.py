#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from aiaccel.config import prepare_config
from aiaccel.job.utils import split_tasks
from rich.progress import track

import soundfile as sf


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("split", type=str, default="train")
    args = parser.parse_args()

    config = prepare_config(
        dataset_path / "config.yaml",
        print_config=True,
    )

    split_path = dataset_path / args.split
    dst_path = split_path / "mix"
    dst_path.mkdir(exist_ok=True)

    src_filename_list = list((split_path / "mix-clean").glob("*.wav"))
    src_filename_list.sort()

    src_filename_list = split_tasks(src_filename_list)

    for src_filename in track(src_filename_list):
        wav, sr = sf.read(src_filename)

        wav = wav[:, : config.n_mics]
        wav += np.sqrt(np.mean(wav**2)) * 10 ** (-config.snr / 20) * np.random.randn(*wav.shape)

        assert np.all(np.abs(wav) < 1)

        sf.write(dst_path / src_filename.name, wav, sr, "PCM_24")


if __name__ == "__main__":
    main()
