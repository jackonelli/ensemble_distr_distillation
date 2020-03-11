"""Loss module"""
import torch
import numpy as np
import torch.distributions.multivariate_normal as torch_mvn

import logging

LOGGER = logging.getLogger(__name__)


def cross_entropy_soft_targets(predicted_distribution, target_distribution):
    """Cross entropy loss with soft targets.
    B = batch size, D = dimension of target (num classes), N = ensemble size

    Args:
        inputs (torch.tensor((B, D - 1))): predicted distribution
        soft_target (torch.tensor((B, D - 1))): target distribution
    """

    return torch.mean(-target_distribution * torch.log(predicted_distribution))


def gaussian_neg_log_likelihood(parameters, target):
    """Negative log likelihood loss for the Gaussian distribution
    B = batch size, D = dimension of target (num classes), N = ensemble size

    Args:
        parameters (torch.tensor((B, D)), torch.tensor((B, D))):
            mean values and variances of y|x for every x in
            batch.

        target (torch.tensor((B, N, D))): sample from the normal
            distribution, if not an ensemble prediction N=1.
    """
    mean = parameters[0]
    var = parameters[1]

    loss = 0
    for batch_index, (mean_b, cov_b) in enumerate(zip(mean, var)):
        cov_mat_b = torch.diag(cov_b)
        distr = torch_mvn.MultivariateNormal(mean_b, cov_mat_b)

        log_prob = distr.log_prob(target[batch_index, :, :])
        loss -= torch.mean(log_prob) / target.size(0)

    return loss


def mse(mean, target):
    """Mean squared loss (torch built-in wrapper)
    B = batch size, D = dimension of target, N = number of samples

    Args:
        mean (torch.tensor((B, D))):
            mean values of y|x for every x in
            batch (and for every ensemble member).
        target (torch.tensor((B, N, D))): Ground truth sample
            (if not an ensemble prediction N=1.)
    """

    _, N, _ = target.size()
    loss_function = torch.nn.MSELoss(reduction="mean")
    total_loss = 0
    for sample_ind in np.arange(N):
        sample = target[:, sample_ind, :]
        total_loss += loss_function(sample, mean)

    return total_loss / N


def inverse_wishart_neg_log_likelihood(parameters, target):
    """Negative log likelihood loss for the inverse-Wishart distribution
    B = batch size, D = target dimension, N = ensemble size

    Args:
        parameters (torch.tensor((B, D)), torch.tensor((B, 1))):
            diagonal of psi and degrees-of-freedom, nu > D - 1, of the
            inverse-Wishart distribution for every x in batch.
        target (torch.tensor((B, N, D))): variance
            (diagonal of covariance matrix)
            as output by N ensemble members.
            """

    # This should only happen when we only have one target (i.e. N=1)
    if target.dim() == 2:
        target = torch.unsqueeze(target, dim=1)

    psi = parameters[0]
    nu = parameters[1]

    normalizer = 0
    ll = 0
    for i in np.arange(target.size(1)):
        cov_mat = [
            torch.diag(target[b, i, :]) for b in np.arange(target.size(0))
        ]
        cov_mat_det = torch.unsqueeze(torch.stack(
            [torch.det(cov_mat_i) for cov_mat_i in cov_mat], dim=0),
                                      dim=1)

        psi_mat = [torch.diag(psi[b, :]) for b in np.arange(target.size(0))]
        psi_mat_det = torch.unsqueeze(torch.stack(
            [torch.det(psi_mat_i) for psi_mat_i in psi_mat], dim=0),
                                      dim=1)

        normalizer += (-(nu / 2) * torch.log(psi_mat_det) +
                       (nu * target.size(-1) / 2) *
                       torch.log(torch.tensor(2, dtype=torch.float32)) +
                       torch.lgamma(nu / 2) +
                       ((nu - target.size(-1) - 1) / 2) *
                       torch.log(cov_mat_det)) / target.size(
                           1)  # Mean over ensemble
        ll += torch.stack([
            0.5 * torch.trace(torch.inverse(psi_mat_i) * cov_mat_i)
            for psi_mat_i, cov_mat_i in zip(psi_mat, cov_mat)
        ],
                          dim=0) / target.size(1)

    return torch.mean(normalizer + ll)  # Mean over batch
