"""Ensemble class"""
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
