"""Loss module"""
import torch
import torch.nn as nn


def scalar_loss(inputs, soft_targets):
    """I think it might be simpler to just use functions for custom loss
    as long as we only use torch functions we should be ok.
    """
    return torch.sum(-soft_targets * torch.log(inputs))


def dirichlet_neg_log_likelihood(alphas, target_distribution):
    """Negative log likelihood loss for the Dirichlet distribution
    B = batch size, C = num classes
    Ugly L1 loss hack (measuring L1 distance from nll to zero)

    Args:
        alphas (torch.tensor((B, C))): alpha vectors for every x in batch
            alpha_c must be > 0 for all c.
        target_distribution (torch.tensor((B, C))): ensemble distribution
    """
    neg_log_like = torch.sum(-(
        (torch.log(target_distribution) * (alphas - 1.0)).sum(-1) +
        torch.lgamma(alphas.sum(-1)) - torch.lgamma(alphas).sum(-1)))
    tmp_L1_loss = torch.nn.L1Loss()
    return tmp_L1_loss(neg_log_like, torch.zeros(neg_log_like.size()))


def sum_of_squares_bayes_risk(alphas, target_distribution, hard_targets=None, lambda_t=None):

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

    l_var = torch.mul(p_hat, (1 - p_hat) /
                      (strength + 1)).sum(-1).mean()

    if hard_targets is not None:
        alphas_tilde = hard_targets + (1 - hard_targets) * alphas
        l_reg = lambda_t * flat_prior_2(alphas_tilde) / alphas.shape[-1]
    else:
        l_reg = 0

    return l_err + l_var + l_reg


def cross_entropy_bayes_risk(alphas, target_distribution, hard_targets=None, lambda_t=None):

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
    l_err = (target_distribution * (torch.digamma(strength) - torch.digamma(alphas))).sum(-1).mean()

    if hard_targets is not None:
        alphas_tilde = hard_targets + (1 - hard_targets) * alphas
        l_reg = lambda_t * flat_prior_2(alphas_tilde) / alphas.shape[-1]
    else:
        l_reg = 0

    return l_err + l_reg


def type_two_maximum_likelihood(alphas, target_distribution, hard_targets=None, lambda_t=None):

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
    l_err = (target_distribution * (torch.log(strength) - torch.log(alphas))).sum(-1).mean()

    if hard_targets is not None:
        alphas_tilde = hard_targets + (1 - hard_targets) * alphas
        l_reg = lambda_t * flat_prior_2(alphas_tilde) / alphas.shape[-1]
    else:
        l_reg = 0

    return l_err + l_reg


def flat_prior(alphas):
    """KL divergence between Dir(alpha) and Dir(1)"""
    log_numerator = torch.lgamma(alphas.sum(-1))
    log_denominator = torch.lgamma(
        torch.tensor(alphas.size(0),
                     dtype=torch.float)) + torch.lgamma(alphas).prod(-1)
    exp_log_p = torch.digamma(alphas) - torch.digamma(alphas.sum())
    digamma_term = torch.sum((alphas - 1.0) * exp_log_p, dim=-1)
    return torch.sum(log_numerator - log_denominator + digamma_term)


def flat_prior_2(alphas):
    """KL divergence between Dir(alpha) and Dir(1)"""
    log_numerator = torch.lgamma(alphas.sum(-1))
    log_denominator = torch.lgamma(
        torch.tensor(alphas.size(-1),
                     dtype=torch.float)) + torch.lgamma(alphas).prod(-1)
    exp_log_p = torch.digamma(alphas) - torch.digamma(alphas.sum(-1, keepdim=True))
    digamma_term = torch.sum((alphas - 1.0) * exp_log_p, dim=-1)
    return torch.mean(log_numerator - log_denominator + digamma_term)

