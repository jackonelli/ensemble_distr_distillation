"""Loss module"""
import torch
import torch.nn as nn


class CrossEntropyLossOneHot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, soft_targets):
        """TODO: extract convergence measure"""
        ctx.save_for_backward(inputs, soft_targets)
        return torch.sum(-soft_targets *
                         nn.functional.log_softmax(inputs, dim=0),
                         dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, soft_targets = ctx.saved_tensors

        return -soft_targets / inputs, None


class CrossEntropyLossOneHotMatrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, soft_targets):
        ctx.save_for_backward(inputs, soft_targets)
        x = -soft_targets * nn.functional.log_softmax(inputs, dim=-1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        inputs, soft_targets = ctx.saved_tensors
        diag_mat = torch.eye(inputs.shape[1])

        grad_input = list()
        for i, inp in enumerate(inputs):
            grad_input.append(grad_output[i, :] * soft_targets[i, :] *
                              (diag_mat - nn.functional.softmax(inp)))

        grad_input = torch.stack(grad_input)

        return grad_input, None
