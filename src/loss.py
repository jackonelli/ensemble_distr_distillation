"""Loss module"""
import torch
import torch.nn as nn


def scalar_loss(inputs, soft_targets):
    """I think it might be simpler to just use functions for custom loss
    as long as we only use torch functions we should be ok.
    """
    return torch.sum(-soft_targets * nn.functional.log_softmax(inputs, dim=0))


class CrossEntropyLossOneHot(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, soft_targets):
        """TODO: extract convergence measure"""
        ctx.save_for_backward(inputs, soft_targets)
        return torch.sum(-soft_targets *
                         nn.functional.log_softmax(inputs, dim=0))

    @staticmethod
    def backward(ctx, grad_output):
        inputs, soft_targets = ctx.saved_tensors

        return -soft_targets / inputs, None
