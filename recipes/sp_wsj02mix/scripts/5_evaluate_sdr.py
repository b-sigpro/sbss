#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path
import pickle as pkl
import warnings

from omegaconf import DictConfig

import numpy as np

from aiaccel.config import load_config, overwrite_omegaconf_dumper
from mir_eval.separation import bss_eval_sources

import soundfile as sf

warnings.simplefilter(action="ignore", category=FutureWarning)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def evaluate_sdr(est_filename: Path, src_path: Path):
    name = est_filename.name

    wav_est = sf.read(est_filename)[0].T
    wav_est = wav_est[np.argsort(np.mean(wav_est**2, axis=1))[-2:]]

    wav_src = np.stack([sf.read(src_path / f"s{sidx}-dry" / name, always_2d=True)[0][:, 0] for sidx in (1, 2)], axis=0)

    T = wav_src.shape[1]
    wav_est = wav_est[:, :T]
    wav_est = np.pad(wav_est, ((0, 0), (0, T - wav_est.shape[1])), mode="constant")

    try:
        sdr, sir, sar, _ = bss_eval_sources(wav_src, wav_est)
    except ValueError as err:
        print(f"Value Error: {err}")

        nans = np.full(wav_est.shape[0], np.nan)

        return name, (nans.copy(), nans.copy(), nans.copy())
    else:
        return name, (sdr, sir, sar)


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    overwrite_omegaconf_dumper()
    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore

    parser = ArgumentParser(description="SDR evaluation")
    parser.add_argument("split", type=str)
    parser.add_argument("inference_name", type=str)

    args = parser.parse_args()

    sep_path = dataset_path / args.inference_name / args.split
    src_path = dataset_path / args.split

    est_filename_list = list(filter(lambda p: p.name not in config.blacklist, sep_path.glob("*.wav")))

    evaluate_sdr_ = partial(evaluate_sdr, src_path=src_path)

    with Pool() as p:
        scores = {}
        for idx, (key, values) in enumerate(p.imap_unordered(evaluate_sdr_, est_filename_list)):
            scores[key] = values

            utt_name = f"{key}"
            if not np.isnan(values[0]).any():
                print(f"{utt_name:>42s} ({idx:04d}) | " + ", ".join(map("{:+06.2f}".format, values[0])))
            else:
                print(f"{utt_name:>42s} ({idx:04d}) | nan error")

    print(f"Average sdr: {np.mean([sdr for sdr, _, _ in scores.values()])}")

    with open(sep_path / "sdr.pkl", "wb") as f:
        pkl.dump(scores, f)


if __name__ == "__main__":
    main()
