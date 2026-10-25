#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from aiaccel.config import prepare_config


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    config = prepare_config(
        dataset_path / "config.yaml",
        print_config=True,
    )

    sp_wsj02mix_path = Path(config.path.sp_wsj02mix)
    wsj02mix_path = Path(config.path.wsj02mix)

    for split in ["tr", "cv", "tt"]:
        split_path = dataset_path / split
        split_path.mkdir(exist_ok=True)

        for src in ["s1", "s2"]:
            (split_path / src).symlink_to(sp_wsj02mix_path / split / src)
            (split_path / f"{src}-dry").symlink_to(wsj02mix_path / split / src)

        (split_path / "mix-clean").symlink_to(sp_wsj02mix_path / split / "mix")


if __name__ == "__main__":
    main()
