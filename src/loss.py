import torch
import torch.nn as nn


class CrossEntropyLossOneHot(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, soft_targets):
        ctx.save_for_backward(inputs, soft_targets)
        x = - soft_targets * nn.functional.log_softmax(inputs, dim=-1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        inputs, soft_targets = ctx.saved_tensors
        grad_input = grad_output * soft_targets

        return grad_input
