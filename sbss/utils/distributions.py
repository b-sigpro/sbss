# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch
from torch.distributions.exp_family import ExponentialFamily


class ApproxBernoulli(torch.distributions.RelaxedBernoulli):
    def rsample(self, sample_shape=torch.Size()):  # noqa
        x: torch.Tensor = super().rsample(sample_shape)  # type: ignore
        return x - x.detach() + (x > 0.5).to(x.dtype)


def cmvlgamma(nu, M):
    lp = M * (M - 1) / 2 * torch.log(torch.tensor(torch.pi)) + torch.lgamma(
        nu[..., None] - torch.arange(M, device="cuda")
    ).sum(-1)
    return lp.to("cuda")


class ComplexWishart(ExponentialFamily):
    def __init__(self, nu, covariance_matrix):
        self.nu = torch.as_tensor(nu, dtype=torch.float32, device=covariance_matrix.device)

        self.covariance_matrix = covariance_matrix
        self.M = covariance_matrix.shape[-1]

    def log_prob(self, Sig):
        Psi = self.covariance_matrix
        Psii = torch.linalg.inv(Psi)
        trPsiiSig = torch.einsum("...mn,...nm->...", Psii, Sig).real.clip(min=1e-6)

        logp = (self.nu - self.M) * torch.linalg.slogdet(Sig)[1]
        logp -= self.nu * torch.linalg.slogdet(Psi)[1]
        logp -= trPsiiSig
        logp -= cmvlgamma(self.nu, self.M)

        return logp


class ComplexInverseWishart(ExponentialFamily):
    def __init__(self, nu, covariance_matrix):
        self.nu = torch.as_tensor(nu, dtype=torch.float32, device=covariance_matrix.device)

        self.covariance_matrix = covariance_matrix
        self.M = covariance_matrix.shape[-1]

    def log_prob(self, Sig):
        Psi = self.covariance_matrix
        Sigi = torch.linalg.inv(Sig)
        trPsiSigi = torch.einsum("...mn,...nm->...", Psi, Sigi).real.clip(min=1e-6)

        logp = -(self.nu + self.M) * torch.linalg.slogdet(Sig)[1]
        logp += self.nu * torch.linalg.slogdet(Psi)[1]
        logp -= trPsiSigi
        logp -= cmvlgamma(self.nu, self.M)

        return logp


class FastComplexWishart(ComplexWishart):
    def log_prob(self, Sig):
        Psi = self.covariance_matrix
        Psii = torch.linalg.inv(Psi)
        trPsiiSig = torch.einsum("...mn,...nm->...", Psii, Sig).real.clip(min=1e-6)

        logp = (self.nu - self.M) * torch.linalg.slogdet(Sig)[1]
        logp -= trPsiiSig

        return logp


class FastComplexInverseWishart(ComplexInverseWishart):
    def __init__(self, nu, covariance_matrix):
        self.nu = torch.as_tensor(nu, dtype=torch.float32, device=covariance_matrix.device)

        self.covariance_matrix = covariance_matrix
        self.M = covariance_matrix.shape[-1]

    def log_prob(self, Sig):
        Psi = self.covariance_matrix
        Sigi = torch.linalg.inv(Sig)
        trPsiSigi = torch.einsum("...mn,...nm->...", Psi, Sigi).real.clip(min=1e-6)

        logp = -(self.nu + self.M) * torch.linalg.slogdet(Sig)[1]
        logp -= trPsiSigi

        return logp
