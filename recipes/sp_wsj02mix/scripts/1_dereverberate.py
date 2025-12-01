#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import os
from pathlib import Path

from omegaconf import DictConfig

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config
import cupy as cp
from gss.wpe import wpe
from rich.progress import track

from librosa.core import istft, stft
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
    dst_path = split_path / "derev"
    dst_path.mkdir(exist_ok=True)

    src_filename_list = list((split_path / "mix").glob("*.wav"))
    src_filename_list.sort()

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        src_filename_list = src_filename_list[start:end]

    for src_filename in track(src_filename_list):
        src_wav, sr = sf.read(src_filename)

        src_spec = stft(src_wav.T, n_fft=config.n_fft, hop_length=config.hop_length)  # [M, F, T]
        src_spec = cp.asarray(src_spec)
        M, F, T = src_spec.shape

        dst_spec = wpe(src_spec.transpose(1, 0, 2), taps=config.wpe.taps, delay=config.wpe.delay)

        dst_wav = istft(dst_spec.get().transpose(1, 0, 2), hop_length=config.hop_length).T

        sf.write(dst_path / src_filename.name, dst_wav, sr, "PCM_24")


if __name__ == "__main__":
    main()
