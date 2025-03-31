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

    sp_wsj02mix_path = Path(config.path.sp_wsj02mix)
    wsj02mix_path = Path(config.path.wsj02mix)

    for split in ["tr", "cv", "tt"]:
        split_path = dataset_path / split
        split_path.mkdir(exist_ok=True)

        for src in ["s1", "s2"]:
            (split_path / src).symlink_to(sp_wsj02mix_path / split / src)
            (split_path / f"{src}-dry").symlink_to(wsj02mix_path / split / src)

        (split_path / "mix-clean").symlink_to(sp_wsj02mix_path / split / "mix")

    done_path = dataset_path / f".{command_name}.done"
    done_path.touch()


if __name__ == "__main__":
    main()
