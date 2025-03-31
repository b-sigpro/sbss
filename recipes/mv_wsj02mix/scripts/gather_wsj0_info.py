#! /usr/bin/env python3

from collections import defaultdict
import json
from pathlib import Path

from omegaconf import DictConfig

from aiaccel.utils import load_config, overwrite_omegaconf_dumper, print_config
from rich.progress import track


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.paren

    overwrite_omegaconf_dumper()

    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    wsj0Nmix_path = {2: Path(config.path.wsj02mix), 3: Path(config.path.wsj03mix)}

    for split in ["tr", "cv", "tt"]:
        split_path = dataset_path / split
        split_path.mkdir(exist_ok=True)

        wsj0_info = defaultdict(set)
        for nspk in [2, 3]:
            print(split, nspk)

            for wav_filename in track((wsj0Nmix_path[nspk] / split / "mix").glob("*.wav")):
                wsj0_names = wav_filename.stem.split("_")[::2]

                for wsj0 in wsj0_names:
                    spk, uttid = wsj0[:3], wsj0[3:]

                    wsj0_info[spk].add(uttid)

        wsj0_info = {spk: sorted(list(wsj0_info[spk])) for spk in sorted(wsj0_info.keys())}

        with open(split_path / "wsj0-info.json", "w") as f:
            json.dump(wsj0_info, f, indent=4)


if __name__ == "__main__":
    main()
