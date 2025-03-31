#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl

from omegaconf import OmegaConf as oc

import numpy as np

from aiaccel.torch.h5py.hdf5_writer import HDF5Writer
from aiaccel.utils import load_config, overwrite_omegaconf_dumper, print_config

import soundfile as sf


class UnsupervisedHDF5Writer(HDF5Writer[Path, None]):
    def __init__(self, split: str, dataset_path: Path, data: str, duration: int, fps: int):
        super().__init__()

        self.split = split
        self.dataset_path = dataset_path

        self.data = data
        self.duration = duration
        self.fps = fps

    def prepare_globals(self) -> tuple[list[Path], None]:
        wav_filename_list = list((self.dataset_path / self.split / self.data).glob("*.wav"))
        wav_filename_list.sort(key=lambda x: x.stat().st_size)

        return wav_filename_list, None

    def prepare_group(
        self,
        item: Path,
        context: None,
    ) -> dict[str, dict[str, np.ndarray]]:
        wav_filename = item

        wav, sr = sf.read(wav_filename)
        tau_rir = int(self.duration / 16000 * self.fps)

        duration, n_mic = wav.shape
        if duration < self.duration:
            return {}

        with open(self.dataset_path / self.split / "arr" / f"{wav_filename.stem}.pkl", "rb") as f:
            mic_pos = np.array(pkl.load(f), dtype=np.float32)

        with open(self.dataset_path / self.split / "traj" / f"{wav_filename.stem}.pkl", "rb") as f:
            spk_pos = np.array(pkl.load(f), dtype=np.float32)

        if spk_pos.shape[1] < tau_rir:
            return {}

        groups = {}
        for tidx, t in enumerate(range(0, duration, self.duration)):
            if t + self.duration > duration:
                t = max(0, duration - self.duration)

            t_rir = int(t / sr * self.fps)
            groups[f"{wav_filename.stem}-{tidx:03d}"] = {
                "wav": wav[t : t + self.duration, :].T,
                "mic_pos": mic_pos,
                "spk_pos": spk_pos[:, t_rir : t_rir + tau_rir],
            }

        return groups


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    # load config
    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("--data", type=str, default="mix")
    parser.add_argument("--duration", type=int, default=64000)
    parser.add_argument("--parallel", action="store_true")
    args, unk_args = parser.parse_known_args()

    base_config = oc.merge(oc.from_cli(unk_args))
    config = load_config(dataset_path / "config.yaml", base_config)
    print_config(config)

    args_str = f"{args.data}_{args.duration}.{args.split}"

    done_filename = dataset_path / f".{command_name}.{args_str}.done"
    done_filename.unlink(missing_ok=True)

    # write HDF5 file
    hdf_filename = dataset_path / "hdf5" / f"unsupervised.{args_str}.hdf5"
    hdf_filename.unlink(missing_ok=True)

    writer = UnsupervisedHDF5Writer(args.split, dataset_path, args.data, args.duration, config.fps)
    writer.write(hdf_filename, parallel=args.parallel)

    done_filename.touch()


if __name__ == "__main__":
    main()
