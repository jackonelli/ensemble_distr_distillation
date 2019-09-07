"""Loss module"""
import torch
import numpy as np


def scalar_loss(inputs, soft_targets):
    """I think it might be simpler to just use functions for custom loss
    as long as we only use torch functions we should be ok.
    """

    return torch.sum(-soft_targets * torch.log(inputs))


def dirichlet_neg_log_likelihood(alphas, target_distribution):
    """Negative log likelihood loss for the Dirichlet distribution
    B = batch size, C = num classes, N = ensemble size
    Ugly L1 loss hack (measuring L1 distance from nll to zero)

    Args:
        alphas (torch.tensor((B, C))): alpha vectors for every x in batch
            alpha_c must be > 0 for all c.
        target_distribution (torch.tensor((B, N, C))): ensemble distribution
            N probability vectors for every data point (B in total).
    """
    ensemble_size = target_distribution.size(1)
    sufficient_statistics = _dirichlet_sufficient_statistics(
        target_distribution)
    inner_alpha_sum = torch.lgamma(alphas.sum(-1))
    outer_alpha_sum = torch.lgamma(alphas).sum(-1)
    log_prob = (sufficient_statistics * (alphas - 1.0)).sum(-1)
    neg_log_like = -ensemble_size * (inner_alpha_sum - outer_alpha_sum +
                                     log_prob)
    tmp_l1_loss = torch.nn.L1Loss()

    return tmp_l1_loss(neg_log_like, torch.zeros(neg_log_like.size()))


def _dirichlet_sufficient_statistics(target_distribution):
    """
    Args:
        target_distribution (torch.tensor((B, N, C))): ensemble distribution
            N probability vectors for every data point (B in total).
    Returns:
        sufficient_statistics (torch.tensor((B, C))):
            Averages over ensemble members
    """
    return torch.mean(torch.log(target_distribution), 1)

def gaussian_neg_log_likelihood(parameters, target, scale=1.0):
    """Negative log likelihood loss for the Gaussian distribution
    B = batch size, D = dimension of target (num classes), N = ensemble size

    Args:
        parameters (torch.tensor((B, N, D)), torch.tensor((B, N, D))):
            mean values and variances of y|x for every x in
            batch (and for every ensemble member).
        target (torch.tensor((B, N, D))): sample from the normal
            distribution, if not an ensemble prediction N=1.
        scale (torch.tensor(B, 1)): scaling parameter for the variance
            (/covariance matrix) for every x in batch.
    """

    mean = parameters[0]
    var = parameters[1]

    normalizer = 0
    ll = 0
    for i in np.arange(mean.size(1)):
        cov_mat = [torch.diag(var[b, i, :]) for b in np.arange(target.size(0))]

        normalizer += torch.stack([0.5 * (target.size(-1) * torch.log(torch.tensor(2 * np.pi))
                                          + torch.log(torch.det(cov_mat_i)))
                                  for cov_mat_i in cov_mat], dim=0)

        ll += torch.sum(torch.stack([0.5 * (target[b, :] - mean[b, i, :]) * (1 / scale) * torch.inverse(cov_mat_i)
                        * torch.transpose((target[b, :] - mean[b, i, :]), 0, -1)
                        for b, cov_mat_i in enumerate(cov_mat)], dim=0), dim=1) / mean.size(1)  # Mean over ensembles

    return torch.mean(normalizer + torch.sum(ll, dim=-1))  # Sum over dimensions, mean over batch


def inverse_wishart_neg_log_likelihood(parameters, target):
    """Negative log likelihood loss for the inverse-Wishart distribution
    B = batch size, D = target dimension, N = ensemble size

    Args:
        parameters (torch.tensor((B, D)), torch.tensor((B, D))):
            diagonal of psi and degrees-of-freedom, nu > D - 1, of the
            inverse-Wishart distribution for every x in batch.
        target (torch.tensor((B, D))): variance (diagonal of covariance matrix)
            as output by N ensemble members.
            """

    psi = parameters[0]
    nu = parameters[1]

    normalizer = 0
    ll = 0
    for i in np.arange(target.size(1)):
        # CAN I DO ANYTHING ABOUT THIS UGLY LIST THING?
        cov_mat = np.stack([torch.diag(target[b, i, :]) for b in np.arange(target.size(0))], axis=0)
        cov_mat_det = np.stack([torch.det(cov_mat_i) for cov_mat_i in cov_mat], axis=0)
        psi_mat = np.stack([torch.diag(psi[b, i, :]) for b in np.arange(target.size(0))], axis=0)
        psi_mat_det = np.stack([torch.det(psi_mat_i) for psi_mat_i in psi_mat], axis=0)

        normalizer += - (nu / 2) * torch.log(psi_mat_det) + (nu * target.size(-1) / 2) * torch.log(2) + torch.lgamma(nu/2)\
            + ((nu + target.size(-1) + 1) / 2) * cov_mat_det
        ll += np.stack([0.5 * torch.trace(psi_mat * torch.inverse(cov_mat_i)) for cov_mat_i in cov_mat], axis=0)

    return normalizer + ll


def gaussian_inv_wishart_neg_log_likelihood(parameters, targets):
    """Negative log likelihood loss for the Gaussian inverse-Wishart distribution
        B = batch size, D = target dimension, N = ensemble size

        Args:
        parameters (torch.tensor((B, 1, D)), torch.tensor((B, 1, D)),
             torch.tensor((B, 1, D)), torch.tensor((B, 1, D)): parameters of the normal distribution (mu_0, scale)
             and of the inverse-Wishart distribution (psi, nu)
        targets (torch.tensor((B, N, D)), torch.tensor((B, N, D)): mean and variance (diagonal of covariance
             matrix) as output by N ensemble members.
        """

    nll_gaussian = gaussian_neg_log_likelihood(parameters[0:2], targets[0])
    nll_inverse_wishart = inverse_wishart_neg_log_likelihood(parameters[2:4], targets[1])

    return nll_gaussian + nll_inverse_wishart


def sum_of_squares_bayes_risk(alphas,
                              target_distribution,
                              hard_targets=None,
                              lambda_t=None):
    """Bayes risk for sum of squares
    B = batch size, C = num classes

    Args:
        alphas (torch.tensor((B, C))): alpha vectors for every x in batch
            alpha_c must be > 0 for all c.
        target_distribution (torch.tensor((B, C))): ensemble distribution
        hard_targets (torch.tensor((B, C))): one-hot-encoded vectors representing
            the true label of every x in batch
        lambda_t (float): weight parameter for regularisation
    """
    strength = torch.sum(alphas, dim=-1, keepdim=True)
    p_hat = torch.div(alphas, strength)
    l_err = torch.nn.MSELoss()(target_distribution, p_hat)

    l_var = torch.mul(p_hat, (1 - p_hat) / (strength + 1)).sum(-1).mean()

    if hard_targets is not None:
        alphas_tilde = hard_targets + (1 - hard_targets) * alphas
        l_reg = lambda_t * flat_prior(alphas_tilde) / alphas.shape[-1]
    else:
        l_reg = 0

    return l_err + l_var + l_reg


def cross_entropy_bayes_risk(alphas,
                             target_distribution,
                             hard_targets=None,
                             lambda_t=None):
    """Bayes risk for cross entropy
        B = batch size, C = num classes

        Args:
            alphas (torch.tensor((B, C))): alpha vectors for every x in batch
                alpha_c must be > 0 for all c.
            target_distribution (torch.tensor((B, C))): ensemble distribution
            hard_targets (torch.tensor((B, C))): one-hot-encoded vectors representing
            the true label of every x in batch
            lambda_t (float): weight parameter for regularisation
        """
    strength = torch.sum(alphas, dim=-1, keepdim=True)
    l_err = (target_distribution *
             (torch.digamma(strength) - torch.digamma(alphas))).sum(-1).mean()

    if hard_targets is not None:
        alphas_tilde = hard_targets + (1 - hard_targets) * alphas
        l_reg = lambda_t * flat_prior(alphas_tilde) / alphas.shape[-1]
    else:
        l_reg = 0

    return l_err + l_reg


def type_two_maximum_likelihood(alphas,
                                target_distribution,
                                hard_targets=None,
                                lambda_t=None):
    """ Type II maximum likelihood
        B = batch size, C = num classes

        Args:
            alphas (torch.tensor((B, C))): alpha vectors for every x in batch
                alpha_c must be > 0 for all c.
            target_distribution (torch.tensor((B, C))): ensemble distribution
            hard_targets (torch.tensor((B, C))): one-hot-encoded vectors representing
            the true label of every x in batch
            lambda_t (float): weight parameter for regularisation
        """
    strength = torch.sum(alphas, dim=-1, keepdim=True)
    l_err = (target_distribution *
             (torch.log(strength) - torch.log(alphas))).sum(-1).mean()

    if hard_targets is not None:
        alphas_tilde = hard_targets + (1 - hard_targets) * alphas
        l_reg = lambda_t * flat_prior(alphas_tilde) / alphas.shape[-1]
    else:
        l_reg = 0

    return l_err + l_reg


def flat_prior(alphas):
    """KL divergence between Dir(alpha) and Dir(1)"""
    log_numerator = torch.lgamma(alphas.sum(-1))
    log_denominator = torch.lgamma(
        torch.tensor(alphas.size(-1),
                     dtype=torch.float)) + torch.lgamma(alphas).prod(-1)
    exp_log_p = torch.digamma(alphas) - torch.digamma(
        alphas.sum(-1, keepdim=True))
    digamma_term = torch.sum((alphas - 1.0) * exp_log_p, dim=-1)
    return torch.mean(log_numerator - log_denominator + digamma_term)
