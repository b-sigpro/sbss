#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from functools import partial
from itertools import permutations
from multiprocessing import Pool
import os
from pathlib import Path
import pickle as pkl

from omegaconf import DictConfig

import numpy as np

from aiaccel.config import load_config, overwrite_omegaconf_dumper

import pystoi
import soundfile as sf

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def evaluate_stoi(est_filename: Path, src_path: Path):
    name = est_filename.name

    wav_est = sf.read(est_filename)[0].T
    wav_est = wav_est[np.argsort(np.mean(wav_est**2, axis=1))[-2:]]

    wav_src = np.stack([sf.read(src_path / f"s{sidx}-dry" / name, always_2d=True)[0][:, 0] for sidx in (1, 2)], axis=0)

    T = wav_src.shape[1]
    wav_est = wav_est[:, :T]
    wav_est = np.pad(wav_est, ((0, 0), (0, T - wav_est.shape[1])), mode="constant")

    max_ave_stoi, max_stois = -np.inf, []
    for order in map(np.asarray, permutations(range(2))):
        stois = np.asarray([pystoi.stoi(src, est, 16000) for src, est in zip(wav_src, wav_est[order])])

        ave_stoi = np.mean(stois)
        if ave_stoi > max_ave_stoi:
            max_ave_stoi, max_stois = ave_stoi, stois

    return name, max_stois


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    overwrite_omegaconf_dumper()
    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore

    parser = ArgumentParser(description="STOI evaluation")
    parser.add_argument("split", type=str)
    parser.add_argument("inference_name", type=str)

    args = parser.parse_args()

    sep_path = dataset_path / args.inference_name / args.split
    src_path = dataset_path / args.split

    est_filename_list = list(filter(lambda p: p.name not in config.blacklist, sep_path.glob("*.wav")))

    evaluate_stoi_ = partial(evaluate_stoi, src_path=src_path)

    with Pool() as p:
        scores = {}
        for idx, (key, values) in enumerate(p.imap_unordered(evaluate_stoi_, est_filename_list)):
            scores[key] = values

            utt_name = f"{key}"
            if not np.isnan(values).any():
                print(f"{utt_name:>42s} ({idx:04d}) | " + ", ".join(map("{:+06.2f}".format, values)))
            else:
                print(f"{utt_name:>42s} ({idx:04d}) | nan error")

    print(f"Average stoi: {np.mean(list(scores.values()))}")

    with open(sep_path / "stoi.pkl", "wb") as f:
        pkl.dump(scores, f)


if __name__ == "__main__":
    main()
