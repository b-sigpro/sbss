# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch.distributions import RelaxedBernoulli, SigmoidTransform
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli


class ApproxBernoulli(RelaxedBernoulli):
    """Approximation of Bernoulli distribution using a relaxed Bernoulli formulation.

    This class provides a straight-through estimator variant of the
    Relaxed Bernoulli distribution by thresholding at 0.5 while
    maintaining gradient flow for reparameterized samples.

    Args:
        temperature (float): Relaxation temperature parameter.
        probs (torch.Tensor, optional): Probability of success.
        logits (torch.Tensor, optional): Log-odds of success.
        validate_args (bool, optional): Whether to validate the input arguments.
    """

    def __init__(self, temperature, probs=None, logits=None, validate_args=None):
        base_dist = LogitRelaxedBernoulli(temperature, probs, logits, validate_args=validate_args)
        super(RelaxedBernoulli, self).__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):  # noqa
        x = super().rsample(sample_shape)
        return x - x.detach() + (x > 0.5).to(x.dtype)


def cmvlgamma(nu, M):
    lp = M * (M - 1) / 2 * torch.log(torch.tensor(torch.pi)) + torch.lgamma(
        nu[..., None] - torch.arange(M, device="cuda")
    ).sum(-1)
    return lp.to("cuda")


class ComplexWishart(ExponentialFamily):
    """Complex Wishart distribution.

    The complex Wishart distribution is a matrix-valued probability distribution
    commonly used to model covariance matrices in the complex domain.

    Args:
        nu (float or torch.Tensor): Degrees of freedom of the distribution.
        covariance_matrix (torch.Tensor): Positive-definite covariance matrix.
    """

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
    """Complex Inverse Wishart distribution.

    The complex inverse Wishart distribution is the conjugate prior
    for complex covariance matrices, often used in Bayesian signal processing.

    Args:
        nu (float or torch.Tensor): Degrees of freedom of the distribution.
        covariance_matrix (torch.Tensor): Positive-definite covariance matrix.
    """

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
    """Fast approximation of Complex Wishart distribution.

    This variant omits constant normalization terms for faster
    computation while preserving dependence on covariance matrices.

    Args:
        nu (float or torch.Tensor): Degrees of freedom of the distribution.
        covariance_matrix (torch.Tensor): Positive-definite covariance matrix.
    """

    def log_prob(self, Sig):
        Psi = self.covariance_matrix
        Psii = torch.linalg.inv(Psi)
        trPsiiSig = torch.einsum("...mn,...nm->...", Psii, Sig).real.clip(min=1e-6)

        logp = (self.nu - self.M) * torch.linalg.slogdet(Sig)[1]
        logp -= trPsiiSig

        return logp


class FastComplexInverseWishart(ComplexInverseWishart):
    """Fast approximation of Complex Inverse Wishart distribution.

    This variant omits normalization constants for efficiency
    while maintaining the primary statistical structure.

    Args:
        nu (float or torch.Tensor): Degrees of freedom of the distribution.
        covariance_matrix (torch.Tensor): Positive-definite covariance matrix.
    """

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
