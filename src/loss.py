"""Loss module"""
import torch
import numpy as np
import torch.distributions.multivariate_normal as torch_mvn
import logging

LOGGER = logging.getLogger(__name__)


def cross_entropy_soft_targets(predictions, soft_targets):
    """Cross entropy loss with soft targets.
    B = batch size, D = dimension of target (num classes), N = ensemble size

    Args:
        inputs (torch.tensor((B, D - 1))): predicted distribution
        soft_target (torch.tensor((B, D - 1))): target distribution
    """

    predicted_distribution = torch.cat(
        (predictions, 1 - torch.sum(predictions, dim=1, keepdim=True)), dim=1)
    target_distribution = torch.cat(
        (soft_targets, 1 - torch.sum(soft_targets, dim=1, keepdim=True)),
        dim=1)

    return torch.sum(-target_distribution * torch.log(predicted_distribution))


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


def gaussian_neg_log_likelihood_1d(parameters, target):
    """Negative log likelihood loss for the Gaussian distribution
    B = batch size, D = dimension of target (always 1), N = ensemble size

    Args:
        parameters (torch.tensor((B, D)), torch.tensor((B, D))):
            mean values and variances of y|x for every x in
            batch.

        target (torch.tensor((B, N, D))): sample from the normal
            distribution, if not an ensemble prediction N=1.
    """
    mean, var = parameters
    target = target.reshape((target.size(0), 1))
    exponent = -0.5 * (target - mean)**2 / var
    log_coeff = -1 / 2 * torch.log(var) - 0.5 * 1 * np.log(2 * np.pi)

    return -(log_coeff + exponent).mean()


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
    mean, var = parameters

    loss = 0
    for batch_index, (mean_b, cov_b) in enumerate(zip(mean, var)):
        cov_mat_b = torch.diag(cov_b)
        distr = torch_mvn.MultivariateNormal(mean_b, cov_mat_b)

        log_prob = distr.log_prob(target[batch_index, :, :])
        loss -= torch.mean(log_prob) / target.size(0)

    return loss


def gaussian_neg_log_likelihood_diag(parameters, target):
    """Negative log likelihood loss for the Gaussian distribution
    B = batch size, D = dimension of target (num classes), N = ensemble size

    Args:
        parameters (torch.tensor((B, D)), torch.tensor((B, D))):
            mean values and variances of y|x for every x in
            batch.

        target (torch.tensor((B, N, D))): sample from the normal
            distribution, if not an ensemble prediction N=1.
    """
    B, N, _ = target.size()
    mu, sigma_sq = parameters

    prec = sigma_sq.pow(-1)

    nll = 0.0
    for mu_b, prec_b, target_b in zip(mu, prec, target):
        sample_var = (target_b - mu_b).var(dim=0)
        trace_term = (prec_b * sample_var).sum() * N / 2
        nll += trace_term - N / 2 * prec_b.prod()

    return nll / B


def _sample_cov(x, mu):
    N, D = x.size()
    diff = (x - mu)
    if D < 2:
        diff = diff.view(1, -1)
    return diff.matmul(diff.t()).squeeze() / (N - 1)


def gaussian_neg_log_likelihood_unopt(parameters, target, scale=None):
    """Negative log likelihood loss for the Gaussian distribution
       This loss is very unoptimized, but it works for when
       we have several sets of parameters
       (i.e. when ensemble members predict parameters)
    B = batch size, D = dimension of target (num classes), N = ensemble size

    Args:
        parameters (torch.tensor((B, N, D)), torch.tensor((B, N, D))):
            mean values and variances of y|x for every x in
            batch.
        target (torch.tensor((B, N, D))): sample from the normal
            distribution, if not an ensemble prediction N=1.
        scale (torch.tensor(B, 1)): scaling parameter for the variance
            (/covariance matrix) for every x in batch.
    """

    B, N, D = target.shape
    mean = parameters[0]
    var = parameters[1]

    average_nll = 0.0
    for b, b_target in enumerate(target):
        cov_mat = torch.diag(var[b])
        diff = b_target - mean[b]
        var_dist = (diff @ torch.inverse(cov_mat) @ diff.T).sum()
        normalizer = 0.5 * torch.log((2 * np.pi)**D * torch.det(cov_mat))
        average_nll += normalizer + var_dist
    average_nll /= B * N

    if average_nll < 0:
        raise Exception("Negative nll")

    return torch.mean(average_nll)  # Mean over batch


def kl_div_gauss_and_mixture_of_gauss(parameters, targets):
    """KL divergence between a single gaussian and a mixture of M gaussians

    for derivation details, see paper.

    Note: The loss is only correct up to a constant w.r.t. the parameters.

    TODO: Support multivarate

    TODO: Support weighted mixture

    B = batch size, N = ensemble size

    Args:
        parameters (torch.tensor((B, 1)), torch.tensor((B, 1))):
            mean values and variances of y|x for every x in
            batch.
        target ((torch.tensor((B, N)), (torch.tensor((B, N)))):
            means and variances of the mixture components
    """

    mu_gauss = parameters[0]
    sigma_sq_gauss = parameters[1]

    mus_mixture = targets[0]
    sigma_sqs_mixture = targets[1]

    mu_bar = mus_mixture.mean(dim=1, keepdim=True)
    term_1 = torch.mean(
        sigma_sqs_mixture +
        (mus_mixture - mu_bar)**2, dim=1, keepdim=True) / sigma_sq_gauss
    term_2 = (mu_bar - mu_gauss)**2
    term_3 = torch.log(sigma_sq_gauss) / 2

    loss = torch.mean(term_1 + term_2 + term_3, dim=0)
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


def efficient_mse(mean, target):
    """Mean squared loss on a matrix form
    B = batch size, D = dimension of target, N = number of samples

    Args:
        mean (torch.tensor((B, D))):
            mean values of y|x for every x in
            batch (and for every ensemble member).
        target (torch.tensor((B, N, D))): Ground truth sample
            (if not an ensemble prediction N=1.)
    """

    # This should only happen when we have only one target (i.e. N=1)
    # TODO: I think I fixed this by always reshaping targets to a 3 dimensions

    if target.dim() == 2:
        target = torch.unsqueeze(target, dim=1)

    if mean.dim() == 2:
        mean = torch.unsqueeze(mean, dim=1)

    ll = torch.mean(torch.diagonal(torch.matmul((target - mean),
                                                torch.transpose(
                                                    (target - mean), -2, -1)),
                                   dim1=-1,
                                   dim2=-2),
                    dim=[0, 1])


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
        # CAN I DO ANYTHING ABOUT THIS UGLY LIST THING?
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


def norm_inv_wish_nll(parameters, targets):
    """NLL loss for the Gaussian inverse-Wishart distribution:

    https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution

    B = batch size, D = target dimension, N = ensemble size

    Args:

        parameters (torch.tensor((B, D)), torch.Variable,
             torch.tensor((B, D)), torch.Variable:

             Tuple order: (mu_0, lambda, psi, nu)
             parameters of the normal distribution (mu_0, lambda)
             and of the inverse-Wishart distribution (psi, nu)

        targets (torch.tensor((B, N, D)), torch.tensor((B, N, D))):
             mean and variance (diagonal of covariance
             matrix) as output by N ensemble members.
    """

    mu_0 = parameters[0]
    lambda_ = parameters[1]
    psi = parameters[2]
    nu = parameters[3]

    mu = targets[0]
    var_diag = targets[1]

    print("mu_0:", mu_0.shape)
    print("mu:", mu.shape)
    print("var_diag:", var_diag.shape)

    nll_gaussian = gaussian_neg_log_likelihood((mu_0, var_diag / lambda_), mu)
    nll_inverse_wishart = inverse_wishart_neg_log_likelihood((psi, nu),
                                                             var_diag)

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
