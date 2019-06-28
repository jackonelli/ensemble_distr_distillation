"""Metrics"""
import torch


def entropy(predicted_distribution):
    """Labels as one hot vectors"""
    return -torch.sum(
        predicted_distribution * torch.log(predicted_distribution))


def nll(true_labels, predicted_distribution):
    """Labels as one hot vectors"""
    true_labels_float = true_labels.float()
    return -torch.sum(true_labels_float * torch.log(predicted_distribution))
