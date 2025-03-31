#! /usr/bin/env python3

from pathlib import Path

from omegaconf import DictConfig

from aiaccel.utils import load_config, overwrite_omegaconf_dumper, print_config


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    overwrite_omegaconf_dumper()

    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    wsj0_path = Path(config.path.wsj0)

    local_wsj0_path = dataset_path / "wsj0"
    local_wsj0_path.mkdir(exist_ok=True)

    for split in ["si_tr_s", "si_dt_05", "si_et_05"]:
        for spk_path in (wsj0_path / split).glob("*"):
            (local_wsj0_path / spk_path.name).symlink_to(spk_path)

    done_path = dataset_path / f".{command_name}.done"
    done_path.touch()


if __name__ == "__main__":
    main()
