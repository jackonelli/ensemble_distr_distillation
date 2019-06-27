import torch
import torch.nn as nn

class CrossEntropyLossOneHot(torch.autograd.Function):

    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()

    def forward(self, x, target):
        # Target should have the same dimension as x
        x = torch.sum(target * nn.functional.softmax(x, dim=-1))
        return x

    #def backward(self, grad_output):
    # Eventuellt att vi behöver denna, ska kolla på det



