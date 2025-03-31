#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from aiaccel.torch.h5py.hdf5_writer import HDF5Writer
from aiaccel.utils import overwrite_omegaconf_dumper

import soundfile as sf


class UnsupervisedHDF5Writer(HDF5Writer[Path, None]):
    def __init__(self, split: str, dataset_path: Path, data: str, duration: int):
        super().__init__()

        self.split = split
        self.dataset_path = dataset_path

        self.data = data
        self.duration = duration

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

        duration, n_mic = wav.shape
        if duration < self.duration:
            return {}

        groups = {}
        for tidx, t in enumerate(range(0, duration, self.duration)):
            if t + self.duration > duration:
                t = max(0, duration - self.duration)

            groups[f"{wav_filename.name}-{tidx:03d}"] = {"wav": wav[t : t + self.duration, :].T}

        return groups


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    # load config
    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("--data", type=str, default="derev")
    parser.add_argument("--duration", type=int, default=64000)
    parser.add_argument("--parallel", action="store_true")
    args, unk_args = parser.parse_known_args()

    args_str = f"{args.data}_{args.duration}.{args.split}"

    # write HDF5 file
    hdf_filename = dataset_path / "hdf5" / f"unsupervised.{args_str}.hdf5"
    hdf_filename.unlink(missing_ok=True)

    writer = UnsupervisedHDF5Writer(args.split, dataset_path, args.data, args.duration)
    writer.write(hdf_filename, parallel=args.parallel)


if __name__ == "__main__":
    main()
