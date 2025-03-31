# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch


def sqrth_inv(x, eps=1e-6):
    ll, v = torch.linalg.eigh(x)
    sqv = torch.sqrt(ll.clip(eps))

    y = torch.einsum("...mn,...n,...on->...mo", v, sqv, v.conj())
    yi = torch.einsum("...mn,...n,...on->...mo", v, 1 / sqv, v.conj())

    return y, yi


def sqrth(x, eps=1e-6):
    ll, v = torch.linalg.eigh(x)
    y = torch.einsum("...mn,...n,...on->...mo", v, torch.sqrt(ll.clip(eps)), v.conj())

    return y
