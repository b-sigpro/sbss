#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import json
import os
from pathlib import Path
import pickle as pkl

from omegaconf import DictConfig

import matplotlib.pyplot as plt
import numpy as np

from aiaccel.utils import load_config, overwrite_omegaconf_dumper, print_config
from mpi4py.futures import MPIPoolExecutor
from rich.progress import track

import soundfile as sf


def sample_condition(idx, split: str, wsj0_info: dict[str, list[str]], config: DictConfig, plot_traj: bool):
    np.random.seed(idx)

    cond = {}

    # sample room conditions
    cond["room_size"] = np.random.uniform(config.min_room_size, config.max_room_size)

    cond["reverb_time"] = np.random.uniform(config.min_rt, config.max_rt)
    cond["snr"] = 30

    # sample mic positions
    r_mic = np.random.uniform(config.min_mic_r, config.max_mic_r)
    mic0 = _sample_from_sphere(r_mic, r_mic)
    for _ in range(10000):
        cond["mic_positions"] = np.stack(
            [mic0, -mic0] + [_sample_from_sphere(r_mic) for m in range(config.n_mic - 2)], axis=0
        )
        cond["mic_positions"] -= np.mean(cond["mic_positions"], axis=0)

        d = np.linalg.norm(cond["mic_positions"][:, None] - cond["mic_positions"][None, :], axis=-1)
        if np.all(d + (config.min_mic_dist + 1) * np.eye(config.n_mic) > config.min_mic_dist):
            break
    else:
        raise Exception("Failed to mic positions")

    # sample array location
    cond["array_location"] = np.empty(3)
    cond["array_location"][:2] = cond["room_size"][:2] / 2 + _sample_from_sphere(config.max_arr_r, n_dim=2)
    cond["array_location"][2] = np.random.uniform(config.min_arr_z, config.max_arr_z)

    # sample source conditions
    cond["sources"] = []
    for _ in range(np.random.choice(config.n_src_list)):
        src = {}

        spk = np.random.choice(list(wsj0_info.keys()))
        uttid = np.random.choice(list(wsj0_info[spk]))
        src["wav_name"] = f"{spk}{uttid}"
        src["snr"] = np.random.uniform(-config.src_snr_range, config.src_snr_range)

        cond["sources"].append(src)

    # calc minimum length for RIR
    n_rirs = np.inf
    for src in cond["sources"]:
        info = sf.info(f"./wsj0/{src["wav_name"][:3]}/{src["wav_name"]}.wav")
        n_rirs_ = int(np.ceil(info.duration * config.fps))
        if n_rirs_ < n_rirs:
            n_rirs = n_rirs_

    # sample trajections of the sources
    for _ in range(10000):
        locations = []
        for _ in range(len(cond["sources"])):
            locations.append(cond["array_location"] + _sample_traj_arc(n_rirs, config))

        if _validate_constraint(locations, cond["array_location"], cond["room_size"], config):
            for src, loc in zip(cond["sources"], locations, strict=False):
                src["location"] = loc

            break
    else:
        raise Exception("Failed to source locations")

    cond_name = "_".join([f"{src["wav_name"]}_{src["snr"]:.3f}" for src in cond["sources"]])

    if plot_traj:
        src_pos1, src_pos2, mic_pos = (
            cond["sources"][0]["location"],
            cond["sources"][1]["location"],
            cond["array_location"],
        )

        os.makedirs(f"./traj_plot/{split}", exist_ok=True)

        fig = plt.figure()
        ax = fig.gca()

        ax.set_title(f"n_rirs: {n_rirs}")

        ax.scatter(src_pos1[:, 0], src_pos1[:, 1], alpha=np.linspace(0.01, 1, src_pos1.shape[0]))
        ax.scatter(src_pos2[:, 0], src_pos2[:, 1], alpha=np.linspace(0.01, 1, src_pos2.shape[0]))
        ax.scatter(mic_pos[0], mic_pos[1])

        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)

        ax.set_aspect("equal")

        fig.tight_layout(pad=0.1)
        fig.savefig(f"./traj_plot/{split}/{cond_name}.png")

        fig.clf()
        plt.close()

    return cond_name, cond


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

    np.random.seed(0)

    split_path = dataset_path / args.split

    with open(split_path / "wsj0-info.json") as f:
        wsj0_info = json.load(f)

    conditions = []

    with MPIPoolExecutor() as pool:
        n_mix = config.n_mix_dict[args.split]
        sample_condition_ = partial(
            sample_condition,
            split=args.split,
            wsj0_info=wsj0_info,
            config=config,
            plot_traj=args.plot_traj,
        )

        for cond_name, cond in track(pool.map(sample_condition_, range(n_mix)), total=n_mix):
            conditions.append((cond_name, cond))

    with open(split_path / "conditions.pkl", "wb") as f:
        pkl.dump(conditions, f)


def _sample_from_sphere(r, r0=0.0, n_dim=3):
    r = np.random.uniform(r0**n_dim, r**n_dim) ** (1 / n_dim)
    e = np.random.randn(n_dim)

    return r * e / np.linalg.norm(e)


def _sample_traj_arc(n_rirs, config: DictConfig):
    # start_p = sample_from_sphere(config.max_src_r, config.min_src_r)
    start_p = np.empty(3)
    start_p[:2] = _sample_from_sphere(config.max_src_r, config.min_src_r, n_dim=2)
    start_p[2] = np.random.uniform(-0.5, 0.5)

    r = np.linalg.norm(start_p[:2])
    omega = np.random.uniform(config.min_omega, config.max_omega)
    sign = np.random.choice([-1, 1], 1)

    rad = sign * np.arange(n_rirs) * omega / config.fps
    rad += np.arctan2(start_p[1], start_p[0])

    traj = np.stack([r * np.cos(rad), r * np.sin(rad), np.full([n_rirs], start_p[2])], axis=1)

    return traj


def _validate_constraint(locations, array_location, room_size, config: DictConfig):
    # far from wall
    for location in locations:
        if np.any(location < config.src2wall_dist) or np.any(location > room_size - config.src2wall_dist):
            return False

    # minimum angular
    r_locations = np.stack(locations) - array_location  # [N, T, 3]
    v = r_locations / np.linalg.norm(r_locations, axis=-1, keepdims=True)

    ang_diff = np.degrees(np.arccos(np.einsum("td,td->t", v[0], v[1])))
    if np.min(ang_diff) < 45:
        return False

    return True


if __name__ == "__main__":
    main()
