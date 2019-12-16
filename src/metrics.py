"""Metrics"""
import logging
import torch
import numpy as np
import src.utils as utils

LOGGER = logging.getLogger(__name__)


class Metric:
    """Metric class"""
    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.running_value = 0.0
        self.counter = 0
        self.memory = []  # So that we can go back an look at the data


    def __str__(self):
        return "{}: {}".format(self.name, self.mean())

    def update(self, labels, outputs):
        self.running_value += self.function(outputs, labels).detach()  # Do this to save memory
        self.counter += 1

    def mean(self):
        if self.counter > 0:
            mean = self.running_value / self.counter
        else:
            mean = float("nan")
            LOGGER.warning("Trying to calculate mean on unpopulated metric.")
        return mean

    def reset(self):
        self.memory.append(self.mean())
        self.running_value = 0.0
        self.counter = 0


def entropy(predicted_distribution, true_labels):
    """Entropy

    B = batch size, C = num classes
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values
    The true labels argument is simply there for conformity
    so that the entropy metric functions like any metric.

    Args:
        NOT USED true_labels: torch.tensor((B, C))
        predicted_distribution: torch.tensor((B, C))

    Returns:
        entropy(ies): torch.tensor(B,)
    """

    return -torch.sum(
        predicted_distribution * torch.log(predicted_distribution), dim=-1)


def uncertainty_separation_parametric(mu, var):
    """Total, epistemic and aleatoric uncertainty

    based on a parametric (normal) variance measure

    M = length of input data x, N = number of distributions

    Args:
        mu: torch.tensor((M, N)): E(y|x) for M x and N distr.
        var: torch.tensor((M, N)): var(y|x) for M x and N distr.

    Returns:
        aleatoric_uncertainty: torch.tensor((M)):
            E_theta[var(y|x, theta)] for M x and N distr.
        epistemic_uncertainty: torch.tensor((M)):
            var_theta[E(y|x, theta)] for M x and N distr.
    """
    epistemic_uncertainty = torch.var(mu, dim=1)
    aleatoric_uncertainty = torch.mean(var, dim=1)
    return aleatoric_uncertainty, epistemic_uncertainty


def uncertainty_separation_variance(predicted_distribution, true_labels):
    """Total, epistemic and aleatoric uncertainty based on a variance measure

    B = batch size, N = num predictions
    Note: if a batch with B samples is given,
    then the output is a tensor with B values
    The true labels argument is simply there for conformity
    so that the entropy metric functions like any metric.

    Args:
        NOT USED true_labels: torch.tensor((B, 1))
        predicted_distribution: torch.tensor((B, N, 2))

    Returns:
        Tuple of uncertainties (relative the maximum uncertainty):
        Total uncertainty: torch.tensor(B,)
        Epistemic uncertainty: torch.tensor(B,)
        Aleatoric uncertainty: torch.tensor(B,)
    """

    total_uncertainty = np.var(predicted_distribution[:, :, 0], axis=-1)
    aleatoric_uncertainty = np.mean(predicted_distribution[:, :, 1], axis=-1)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty


def uncertainty_separation_entropy(predicted_distribution, true_labels):
    """Total, epistemic and aleatoric uncertainty based on an entropy measure

    B = batch size, C = num classes, N = num predictions
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values
    The true labels argument is simply there for conformity
    so that the entropy metric functions like any metric.
    in the same context as the other metrices?

    Args:
        NOT USED true_labels: torch.tensor((B, C))
        predicted_distribution: torch.tensor((B, N, C))

    Returns:
        Tuple of uncertainties (relative the maximum uncertainty):
        Total uncertainty: torch.tensor(B,)
        Epistemic uncertainty: torch.tensor(B,)
        Aleatoric uncertainty: torch.tensor(B,)
    """

    # We calculate the uncertainties relative the maximum possible uncertainty:
    # (log(C))
    max_entropy = torch.log(
        torch.tensor(predicted_distribution.size(-1)).float())

    mean_predictions = torch.mean(predicted_distribution, dim=1)
    total_uncertainty = -torch.sum(
        mean_predictions * torch.log(mean_predictions), dim=-1) / max_entropy
    aleatoric_uncertainty = - torch.sum(predicted_distribution * torch.log(predicted_distribution), dim=[1, 2]) \
        / (max_entropy * predicted_distribution.size(1))
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty


def nll(predicted_distribution, true_labels):
    """Negative log likelihood

    B = batch size, C = num classes
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values

    Args:
        true_labels: torch.tensor((B, C))
        predicted_distribution: torch.tensor((B, C))

    Returns:
        nll(s): torch.tensor(B,)
    """

    true_labels_float = true_labels.float()
    return -torch.sum(true_labels_float * torch.log(predicted_distribution),
                      dim=-1)


def brier_score(predicted_distribution, true_labels):
    """Brier score

    B = batch size, C = num classes
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values

    Args:
        true_labels: torch.tensor((B, C))
        predicted_distribution: torch.tensor((B, C))

    Returns:
        Brier score(s): torch.tensor(B,)
    """
    true_labels_float = true_labels.float()


def accuracy(predicted_distribution, true_labels):
    """ Accuracy
    B = batch size
    K = number of classes

    Args:
        true_labels: torch.tensor(B)
        predicted_distribution: torch.tensor(B, K)

    Returns:
        Accuracy: float
    """
    predicted_labels, _ = utils.tensor_argmax(predicted_distribution)
    number_of_elements = np.prod(true_labels.size())

    if number_of_elements == 0:
        number_of_elements = 1
    return (true_labels == predicted_labels.int()
            ).sum().item() / number_of_elements


def accuracy_soft_labels(predicted_distribution, target_distribution):
    """ Accuracy
    B = batch size
    K = number of classes

    Args:
        target_distribution: torch.tensor(B, K-1)
        predicted_distribution: torch.tensor(B, K-1)

    Returns:
        Accuracy: float
    """

    predicted_distribution = torch.cat(
        (predicted_distribution,
         1 - torch.sum(predicted_distribution, dim=1, keepdim=True)),
        dim=1)
    target_distribution = torch.cat(
        (target_distribution,
         1 - torch.sum(target_distribution, dim=1, keepdim=True)),
        dim=1)

    predicted_labels, _ = utils.tensor_argmax(predicted_distribution)
    target_labels, _ = utils.tensor_argmax(target_distribution)
    number_of_elements = np.prod(target_labels.size(0))

    if number_of_elements == 0:
        number_of_elements = 1
    return (target_labels.int() == predicted_labels.int()
            ).sum().item() / number_of_elements


def accuracy_logits(predicted_logits, targets, label_targets=False):
    """ Accuracy given that the inputs are logits assumed to be scaled relative the last class K
    B = batch size
    K = number of classes
    N = number of ensemble member

    Args:
        targets: torch.tensor(B, N, K-1) if logits targets, (B, K) otherwise
        predicted_logits: torch.tensor(B, (N,) K-1)
        softmax_targets: specifies if the targets is in logits or in labels form

    Returns:
        Accuracy: float
    """
    number_of_elements = np.prod(predicted_logits.size(0))
    if predicted_logits.dim() == 3:
        predicted_distribution = torch.mean((torch.nn.Softmax(dim=-1))(torch.cat((predicted_logits,
                                                                                  torch.zeros(number_of_elements,
                                                                                              predicted_logits.size(1), 1)),
                                                                                 dim=-1)), dim=1)
    else:
        predicted_distribution = (torch.nn.Softmax(dim=-1))(torch.cat((predicted_logits,
                                                                       torch.zeros(number_of_elements, 1)), dim=-1))

    if label_targets:
        target_labels = targets

    else:
        target_distribution = torch.mean((torch.nn.Softmax(dim=-1))(torch.cat((targets,
                                                                    torch.zeros(number_of_elements,
                                                                                targets.size(1), 1)),  dim=-1)), dim=1)
        target_labels, _ = utils.tensor_argmax(target_distribution)

    predicted_labels, _ = utils.tensor_argmax(predicted_distribution)

    if number_of_elements == 0:
        number_of_elements = 1
    return (target_labels.int() == predicted_labels.int()
            ).sum().item() / number_of_elements


def error(predicted_distribution, true_labels):
    """ Error
    B = batch size

    Args:
        true_labels: torch.tensor(B)
        predicted_distribution: torch.tensor(B)

    Returns:
        Error: float
    """
    predicted_labels = utils.tensor_argmax(predicted_distribution)
    number_of_elements = np.prod(true_labels.size())
    if number_of_elements == 0:
        number_of_elements = 1

    return (true_labels != predicted_labels).sum().item() / number_of_elements


def squared_error(predictions, targets):
    """ Error
    B = batch size
    D = output dimension

    Args:
        targets: torch.tensor(B, D)
        predictions: (torch.tensor(B, 2*D)),
            estimated mean and variances of the
            normal distribution of targets arranged as
            [mean_1, ... mean_D, var_1, ..., var_D]

    Returns:
        Error: float
    """

    number_of_elements = targets.size(0)
    if number_of_elements == 0:
        number_of_elements = 1

    return ((targets - predictions[:, :targets.size(-1)])**
            2).sum().item() / number_of_elements
