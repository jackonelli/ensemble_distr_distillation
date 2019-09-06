"""Metrics"""
import logging
import torch
import numpy as np
<<<<<<< HEAD
import src.utils as utils
=======
import utils
>>>>>>> 4d95c77f1e27473be9140e7c0c7c299fed3ae676

LOGGER = logging.getLogger(__name__)


class Metric:
    """Metric class"""

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.running_value = 0.0
        self.counter = 0

    def __str__(self):
        return "{}: {}".format(self.name, self.mean())

    def update(self, labels, outputs):
        self.running_value += self.function(labels, outputs)
        self.counter += 1

    def mean(self):
        if self.counter > 0:
            mean = self.running_value / self.counter
        else:
            mean = float("nan")
            LOGGER.warning("Trying to calculate mean on unpopulated metric.")
        return mean

    def reset(self):
        self.running_value = 0.0
        self.counter = 0


def entropy(predicted_distribution):
    """Entropy

    B = batch size, C = num classes
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values

    Args:
        NOT USED true_labels: torch.tensor((B, C))
        predicted_distribution: torch.tensor((B, C))

    Returns:
        entropy(ies): torch.tensor(B,)
    """

    return -torch.sum(
        predicted_distribution * torch.log(predicted_distribution), dim=-1)


def nll(true_labels, predicted_distribution):
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


def brier_score(true_labels, predicted_distribution):
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


def accuracy(true_labels, predicted_distribution):
    """ Accuracy
    B = batch size

    Args:
        true_labels: torch.tensor(B)
        predicted_distribution: torch.tensor(B)

    Returns:
        Accuracy: float
    """
    predicted_labels, _ = utils.tensor_argmax(predicted_distribution)
    number_of_elements = np.prod(true_labels.size())
    if number_of_elements == 0:
        number_of_elements = 1
    return (true_labels == predicted_labels).sum().item() / number_of_elements


def error(true_labels, predicted_distribution):
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


def squared_error(targets, predictions):
    """ Error
    B = batch size
    D = output dimension

    Args:
        targets: torch.tensor(B, D)
<<<<<<< HEAD
        predictions: (torch.tensor(B, D), torch.tensor(B, D)), tuple of estimated mean and variances of the
                     normal distribution of targets
=======
        predictions: torch.tensor(B, 2D), vector of estimated mean and variances of the
                     normal distribution of targets arranged as [mean_1, ..., mean_D, var_1, ..., var_D]
>>>>>>> 4d95c77f1e27473be9140e7c0c7c299fed3ae676

    Returns:
        Error: float
    """

<<<<<<< HEAD
    number_of_elements = targets.size(0)
    if number_of_elements == 0:
        number_of_elements = 1

    return ((targets - predictions[0])**2).sum().item() / number_of_elements
=======
    number_of_elements = targets.size()
    if number_of_elements == 0:
        number_of_elements = 1

    return ((targets - predictions[:targets.size()])**2).sum().item() / number_of_elements
>>>>>>> 4d95c77f1e27473be9140e7c0c7c299fed3ae676
