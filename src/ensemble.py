"""Ensemble class"""
import torch
import torch.nn as nn


class Ensemble():
    def __init__(self):
        self.members = list()

    def add_member(self, new_member):
        if issubclass(new_member, nn.Module):
            self.members.append(new_member)
        else:
            raise ValueError("Ensemble member must be nn.Module subclass")

    def add_multiple(self, number_of, constructor):
        for _ in range(number_of):
            self.add_member(constructor())

    def train(self, num_epochs):
        pass

    def prediction(self, x, t=1):

        pred = list()
        for member in self.members:
            pred.append(member.forward(x, t))  # For future use rather

        pred_mean = torch.zeros([x.size[0], self.members[0].output_size], dtype=torch.float32)
        for p in pred:
            pred_mean += (1 / len(self.members)) * p

        return pred_mean

