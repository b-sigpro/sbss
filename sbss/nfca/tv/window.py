# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch


def calc_exponential_window(T: int, decay: float, device: str | torch.device = "cuda"):
    """_summary_

    Args:
        T (int): window length
        decay (float): decay parameter

    Returns:
        Window (torch [T, T])
    """

    win = torch.arange(T)
    win = torch.abs(win[None, :] - win[:, None])
    win = (1 - decay) ** win

    return win.to(device)


def calc_rectangular_window(T: int, decay: float, device: str | torch.device = "cuda"):
    t = torch.arange(T, dtype=torch.float32)

    win = (t[None, :] - t[:, None]).abs().less_equal_(decay)
    win[:decay, 0] += torch.arange(decay, 0, -1)
    win[-decay:, -1] += torch.arange(1, decay + 1)

    return win.to(device)
