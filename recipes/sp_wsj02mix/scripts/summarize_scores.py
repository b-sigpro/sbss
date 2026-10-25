#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl

import numpy as np


def main():
    parser = ArgumentParser()
    parser.add_argument("inference_name", type=str)
    args = parser.parse_args()

    title_texts = []
    score_texts = []
    for split in ["cv", "tt"]:
        # sdr
        title_texts.append("  SDR")

        with open(Path.cwd() / "results" / args.inference_name / split / "sdr.pkl", "rb") as f:
            scores = pkl.load(f)

        avg_sdr = np.mean([sdr for sdr, _, _ in scores.values()])
        score_texts.append(f"{avg_sdr:5.2f}")

        # pesq
        title_texts.append("PESQ")

        with open(Path.cwd() / "results" / args.inference_name / split / "pesq.pkl", "rb") as f:
            scores = pkl.load(f)

        avg_pesq = np.mean(list(scores.values()))
        score_texts.append(f"{avg_pesq:.2f}")

        # stoi
        title_texts.append("STOI")

        with open(Path.cwd() / "results" / args.inference_name / split / "stoi.pkl", "rb") as f:
            scores = pkl.load(f)

        avg_stoi = np.mean(list(scores.values()))
        score_texts.append(f"{avg_stoi:.2f}")

    print(", ".join(title_texts))
    print(", ".join(score_texts))


if __name__ == "__main__":
    main()
