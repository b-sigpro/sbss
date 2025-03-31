#! /usr/bin/env python3

from typing import Any

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path
import pickle as pkl
import sys

from omegaconf import DictConfig

import numpy as np

from aiaccel.utils import load_config, overwrite_omegaconf_dumper, print_config
from rich.console import Console
from rich.progress import track

import soundfile as sf


def generate_mixture(item: tuple[str, dict[str, Any]], split_path: Path, wsj0_path: Path, config: DictConfig):
    import gpuRIR as gr

    cond_name, cond = item

    # generate rirs
    mic_positions = cond["array_location"] + cond["mic_positions"]

    beta = gr.beta_SabineEstimation(cond["room_size"], cond["reverb_time"])
    nb_img = gr.t2n(gr.att2t_SabineEstimator(60, cond["reverb_time"]), cond["room_size"])

    rirs = []
    rirs_dry = []
    for src in cond["sources"]:
        rir = gr.simulateRIR(
            cond["room_size"], beta, src["location"], mic_positions, nb_img, cond["reverb_time"], config.sr
        )  # [S, M, T]
        rirs.append(rir)
        rirs_dry.append(rir[..., : int(0.05 * config.sr)])

    # load wsj0 wavs
    wav_wsj_list = []
    for src in cond["sources"]:
        wav_name = src["wav_name"]
        wav, sr = sf.read(wsj0_path / wav_name[:3] / f"{wav_name}.wav")
        wav_wsj_list.append(wav)

    T = min([len(w) for w in wav_wsj_list])

    # generate source images
    wav_img_list = []
    for sidx, src in enumerate(cond["sources"]):
        wav_wsj = wav_wsj_list[sidx][:T]
        timestamps = np.arange(0, T / sr, 1 / config.fps)
        n_imps = len(timestamps)

        # generate reverberant image
        wav_img = gr.simulateTrajectory(wav_wsj, rirs[sidx][:n_imps], timestamps=timestamps, fs=sr)[:T]
        scale = 0.01 / np.sqrt(np.mean(wav_img**2)) * 10 ** (src["snr"] / 20)
        wav_img *= scale

        assert np.all(np.abs(wav_img) < 1.0)
        sf.write(split_path / f"s{sidx+1}" / f"{cond_name}.wav", wav_img, config.sr, "PCM_24")

        wav_img_list.append(wav_img)

        # generate dry image
        wav_dry = gr.simulateTrajectory(wav_wsj, rirs_dry[sidx][:n_imps], timestamps=timestamps, fs=sr)[:T] * scale

        assert np.all(np.abs(wav_dry) < 1.0)
        sf.write(split_path / f"s{sidx+1}-dry" / f"{cond_name}.wav", wav_dry, config.sr, "PCM_24")

    with open(split_path / "arr" / f"{cond_name}.pkl", mode="wb") as f:
        pkl.dump(cond["mic_positions"], f)

    src_traj = np.stack([src["location"] - cond["array_location"] for src in cond["sources"]])

    with open(split_path / "traj" / f"{cond_name}.pkl", mode="wb") as f:
        pkl.dump(src_traj, f)

    # generate mixture signal
    wav_mix = np.sum(wav_img_list, axis=0)

    noi_std = np.sqrt(np.mean(wav_mix**2)) * 10 ** (-cond["snr"] / 20)
    wav_mix += np.random.normal(0, noi_std, size=wav_mix.shape)

    assert np.all(np.abs(wav_mix) < 1)
    sf.write(split_path / "mix" / f"{cond_name}.wav", wav_mix, config.sr, "PCM_24")


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("--plot_traj", action="store_true")
    args, unk_args = parser.parse_known_args()

    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    wsj0_path = dataset_path / "wsj0"

    split_path: Path = dataset_path / args.split

    dst_path = split_path / "mix"
    dst_path.mkdir(exist_ok=True)

    for s in range(max(config.n_src_list)):
        (split_path / f"s{s+1}").mkdir(exist_ok=True)
        (split_path / f"s{s+1}-dry").mkdir(exist_ok=True)
        (split_path / "arr").mkdir(exist_ok=True)
        (split_path / "traj").mkdir(exist_ok=True)

    with open(split_path / "conditions.pkl", "rb") as f:
        conditions = pkl.load(f)

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        conditions = [conditions[ii] for ii in range(start, end) if ii < len(conditions)]

    generate_mixture_ = partial(
        generate_mixture,
        split_path=split_path,
        wsj0_path=wsj0_path,
        config=config,
    )

    for _ in track(
        map(generate_mixture_, conditions), total=len(conditions), console=Console(file=sys.stderr, force_terminal=True)
    ):
        pass


if __name__ == "__main__":
    main()
