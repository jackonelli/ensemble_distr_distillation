"""Metrics"""
import torch


def nll(true_labels, predicted_labels):
    """Labels as one hot vectors"""
    true_labels_float = true_labels.float()
    predicted_labels_float = predicted_labels.float()
    return -torch.sum(true_labels_float * torch.log(predicted_labels_float))
