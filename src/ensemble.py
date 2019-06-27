"""Ensemble class"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class EnsembleMember(nn.Module, ABC):
    """Parent class for keeping common logic in one place"""

    def __init__(self, loss_function):
        super().__init__()
        self.loss = loss_function
        self.optimizer = None

    def train(self, train_loader, num_epochs):
        # if self.loss is None or not issubclass(self.loss,
        #                                        nn.modules.loss._Loss):
        #     raise ValueError
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader)
            print("Epoch {}: Loss: {}".format(epoch, loss))

    def train_epoch(self, train_loader):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            loss = self.calculate_loss(inputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def calculate_loss(self, inputs, labels):
        pass


class Ensemble():
    def __init__(self):
        self.members = list()

    def add_member(self, new_member):
        if issubclass(new_member, EnsembleMember):
                new_member, EnsembleMember):
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
            pred.append(member.predict(x, t))  # For future use rather

        pred_mean = torch.zeros([x.size[0], self.members[0].output_size],
                                dtype=torch.float32)
        for p in pred:
            pred_mean += (1 / len(self.members)) * p

        return pred_mean
