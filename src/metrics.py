"""Metrics"""
import logging
import torch
import numpy as np
import utils

LOGGER = logging.getLogger(__name__)


class Metric:
    """Metric class"""

    def __init__(self, label, function):
        self.label = label
        self.function = function
        self.running_value = 0.0
        self.counter = 0

    def __str__(self):
        return "{}: {}".format(self.label, self.running_value)

    def add_sample(self, true_labels, predicted_distribution):
        batch_size = true_labels.size(0)
        sample = self.function(true_labels, predicted_distribution)
        self.update_value(sample, batch_size)

    def update_value(self, value, batch_size=1):
        """In principle a private function but can be used manually"""
        self.running_value += value
        self.counter += batch_size

    def mean(self):
        return self.running_value / self.counter

    def reset(self):
        self.running_value = 0.0
        self.counter = 0


class MetricsDict:
    """Wrapper class for metrics dict
    TODO: Inherit dict instead
    """

    def __init__(self):
        self._dict = dict()
        self.loss = 0.0

    def __str__(self):
        string = "Loss: {}".format(self.loss)
        for metric in self._dict.values():
            string += " {}".format(metric)
        return string

    def __getitem__(self, key):
        item = None
        if key == "loss":
            item = self.loss
        else:
            item = self._dict[key]
        return item

    def pop(self, key):
        if key == "loss":
            item = self.loss
        else:
            item = self._dict.pop(key)
        return item

    def add_by_keys(self, metrics_keys):
        """Helper function for convenient metrics inclusion"""
        if isinstance(metrics_keys, list):
            pass
        elif isinstance(metrics_keys, str):
            metrics_keys = [metrics_keys]
        else:
            LOGGER.error(
                "Keys must be list of strings or simple string, got {}".format(
                    type(metrics_keys)))
        for key in metrics_keys:
            metric_function = AVAILABLE_METRICS.get(key)
            if metric_function:
                if key not in self._dict:
                    LOGGER.info("Adding metric: {}".format(key))
                    self._dict[key] = Metric(label=key,
                                             function=metric_function)
                else:
                    LOGGER.error(
                        "Metric: {} already in metrics dict".format(key))
            else:
                LOGGER.error("Metric: {} not available".format(key))

    def update(self, loss, true_labels, predicted_distribution):
        """To be run at every epoch."""
        self.loss += loss
        for metric in self._dict.values():
            metric.add_sample(true_labels, predicted_distribution)

    def reset(self):
        self.loss = 0.0
        for metric in self._dict.values():
            metric.reset()


def entropy(true_labels, predicted_distribution):
    """Entropy

    B = batch size, C = num classes
    Labels as one hot vectors
    Note: if a batch with B samples is given,
    then the output is a tensor with B values

    Args:
        predicted_distribution: torch.tensor((B, C))

    Returns:
        entropy: float
    """

    return -torch.sum(
        predicted_distribution * torch.log(predicted_distribution))


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
        nll: float
    """

    true_labels_float = true_labels.float()
    return -torch.sum(true_labels_float * torch.log(predicted_distribution))


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
        Brier score: float
    """
    true_labels_float = true_labels.float()
    LOGGER.error("Adding unimplemented Brier score.")
    raise NotImplementedError("Brier score")


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


AVAILABLE_METRICS = {
    "entropy": entropy,
    "accuracy": accuracy,
    "error": error,
    "brier_score": brier_score,
    "nll": nll
}
