# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from einops import rearrange

import torch
from torch import nn

from lightning import LightningModule
from torchaudio.transforms import InverseSpectrogram

import soundfile as sf

from sbss.utils.separator import main


@dataclass
class Context:
    model: LightningModule
    istft: nn.Module
    config: ListConfig | DictConfig


def add_common_args(parser):
    parser.add_argument("--out_ch", type=int, default=0)
    parser.add_argument("--n_iter", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")


def initialize(args: Namespace, unk_args: list[str]):
    with open(args.model_path / "config.pkl", "rb") as f:
        config = pkl.load(f)

    checkpoint_path = args.model_path / "version_0" / "checkpoints" / "last.ckpt"
    config.task._target_ += ".load_from_checkpoint"
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=args.device,
    )
    model.eval()

    istft = InverseSpectrogram(config.n_fft, hop_length=config.hop_length).to(args.device)

    return Context(model, istft, config)


def separate(src_filename: Path, dst_filename: Path, ctx: Context, args: Namespace, unk_args: list[str]):
    model, istft = ctx.model, ctx.istft

    # load wav
    src_wav, sr = sf.read(src_filename, dtype="float32")
    src_wav = rearrange(torch.from_numpy(src_wav).to(model.device), "t m -> 1 m t")

    # calculate spectrogram
    xraw = model.stft(src_wav)[..., : src_wav.shape[-1] // ctx.config.hop_length]  # [B, F, M, T]
    scale = xraw.abs().square().clip(1e-6).mean(dim=(1, 2, 3), keepdims=True).sqrt()
    x = xraw / scale
    _, F, M, T = x.shape

    # encode
    z = model.encoder(x)

    # decode
    lm = model.decoder(z)  # [B, F, N, T]
    lmc = lm.to(torch.complex64)

    # estimate scm
    H = model.scm_estimator(lmc, x, args.n_iter)

    # Wiener filtering
    eI = 1e-6 * torch.eye(M, dtype=x.dtype, device=x.device)
    Yi = torch.linalg.inv(torch.einsum("bfkt,bfkmn->bftmn", lmc, H) + eI)  # [F, T, M, M]
    Yk = torch.einsum("bfkt,bfkn->bftkn", lmc, H[..., args.out_ch, :])  # [T, F, M, M]

    s = torch.einsum("bftkn,bftno,bfot->bkft", Yk, Yi, xraw)
    dst_wav = rearrange(istft(s, src_wav.shape[-1]), "1 m t -> t m")

    # save separated signal
    sf.write(dst_filename, dst_wav.cpu().numpy(), sr, "PCM_24")


if __name__ == "__main__":
    main(add_common_args, initialize, separate)
