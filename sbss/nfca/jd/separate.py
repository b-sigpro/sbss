# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from einops import rearrange
import numpy as np

import torch
from torch import nn
from torch.nn import functional as fn  # noqa

from torchaudio.transforms import InverseSpectrogram

import soundfile as sf

from sbss.utils.separator import main


@dataclass
class Context:
    model: nn.Module
    istft: nn.Module
    config: ListConfig | DictConfig


def add_common_args(parser):
    parser.add_argument("--out_ch", type=int, default=0)
    parser.add_argument("--n_mic", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")


def initialize(args: Namespace, unk_args: list[str]):
    with open(args.model_path / "config.pkl", "rb") as f:
        config = pkl.load(f)

    checkpoint_path = args.model_path / "version_0" / "checkpoints" / "last.ckpt"
    print(checkpoint_path)
    config.task._target_ += ".load_from_checkpoint"
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=args.device,
    )
    model.eval()

    istft = InverseSpectrogram(config.n_fft, hop_length=config.hop_length).to(args.device)

    ctx = Context(model, istft, config)

    return ctx


def separate(src_filename: Path, dst_filename: Path, ctx: Context, args: Namespace, unk_args: list[str]):
    model, istft = ctx.model, ctx.istft

    # load wav
    src_wav, sr = sf.read(src_filename, dtype=np.float32)
    src_wav = rearrange(torch.from_numpy(src_wav).to(model.device), "t m -> 1 m t")

    if args.n_mic is not None:
        channels = src_wav.square().mean(dim=(0, 2)).argsort(descending=True)[: args.n_mic]
        src_wav = src_wav[:, channels]  # 音量でかいnum_channelsのみ

    # calculate spectrogram
    xraw = model.stft(src_wav)[..., : src_wav.shape[-1] // ctx.config.hop_length]  # [B, F, M, T]
    scale = xraw.abs().square().clip(1e-6).mean(dim=(1, 2, 3), keepdims=True).sqrt()
    x = xraw / scale

    # encode
    z, g, Q, xt = model.encoder(x)

    # decode
    lm = model.decoder(z)  # [B, F, N, T]

    # Wiener filtering
    yt = torch.einsum("bfnt,bfmn->bfmt", lm, g).add(1e-6)
    Qx_yt = torch.einsum("bfmn,bfnt->bfmt", Q, xraw) / yt
    s = torch.einsum("bfm,bfnt,bfmn,bfmt->bnft", torch.linalg.inv(Q)[:, :, args.out_ch], lm, g, Qx_yt)

    dst_wav = istft(s, src_wav.shape[-1])
    dst_wav = rearrange(dst_wav, "1 m t -> t m")

    # save separated signal
    sf.write(dst_filename, dst_wav.cpu().numpy(), sr, "PCM_24")


if __name__ == "__main__":
    main(add_common_args, initialize, separate)
