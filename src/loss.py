"""Loss module"""
import torch
import torch.nn as nn


class CrossEntropyLossOneHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, soft_targets):
        """TODO: extract convergence measure"""
        ctx.save_for_backward(inputs, soft_targets)
        return torch.sum(-soft_targets * nn.functional.log_softmax(inputs), -1)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, soft_targets = ctx.saved_tensors

        return -soft_targets / inputs
