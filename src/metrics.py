"""Metrics"""
import torch


def entropy(predicted_distribution):
    """Entropy
    Note: if a batch with N samples is given,
    then the output is a tensor with N values
    """
    return -torch.sum(
        predicted_distribution * torch.log(predicted_distribution), -1)


def nll(true_labels, predicted_distribution):
    """Negative log likelihood
    Labels as one hot vectors
    Note: if a batch with N samples is given,
    then the output is a tensor with N values
    """
    true_labels_float = true_labels.float()
    return -torch.sum(true_labels_float * torch.log(predicted_distribution),
                      -1)


def brier_score(true_labels, predicted_distribution):
    """Labels as one hot vectors"""
    true_labels_float = true_labels.float()
