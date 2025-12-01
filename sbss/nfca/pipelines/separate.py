# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

from einops import rearrange

import torch

from aiaccel.torch.lightning import load_checkpoint
from aiaccel.torch.pipelines import BasePipeline, reorder_fields
import attrs

import soundfile as sf


@attrs.define(slots=False, field_transformer=reorder_fields)
class SeparationPipeline(BasePipeline):
    checkpoint_path: Path
    out_ch: int = 0
    device: str = "cuda"

    overwrite_config: dict[str, Any] | None = None

    src_ext: str = "wav"
    dst_ext: str = "wav"

    def setup(self) -> None:
        self.model, self.config = load_checkpoint(
            self.checkpoint_path,
            device=self.device,
            overwrite_config=self.overwrite_config,
        )
        self.model.eval()

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model(wav, out_ch=self.out_ch)

    @torch.inference_mode()
    def process_one(self, src_filename: Path, dst_filename: Path) -> None:
        wav_mix, sr = sf.read(src_filename, dtype="float32")
        assert sr == self.config.sr, f"Sample rate mismatch: {sr} != {self.config.sr}"

        wav_mix = rearrange(torch.from_numpy(wav_mix), "t m -> 1 m t").to(self.device)
        wav_sep, _ = self(wav_mix)
        wav_sep = wav_sep.squeeze(0).T.cpu().numpy()

        sf.write(dst_filename, wav_sep, sr)

    @classmethod
    def _prepare_parser(cls, fields: list[attrs.Attribute]) -> ArgumentParser:
        return super()._prepare_parser(list(filter(lambda f: f.name != "overwrite_config", fields)))

    @classmethod
    def _process_unk_args(cls, unk_args: list[str], kwargs: dict[str, Any], parser: ArgumentParser) -> dict[str, Any]:
        return kwargs | {"overwrite_config": oc.from_cli(unk_args)}


if __name__ == "__main__":
    SeparationPipeline.main()
