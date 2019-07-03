"""Ensemble class"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import logging
import utils

class EnsembleMember(nn.Module, ABC):
    """Parent class for keeping common logic in one place"""

    def __init__(self, loss_function, device=torch.device('cpu')):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.loss = loss_function
        self.optimizer = None
        self._log.info("Moving model to device: {}".format(device))
        self.device = device

    def train(self, train_loader, num_epochs):
        if self.loss is None or not issubclass(type(self.loss),
                                               nn.modules.loss._Loss):
            raise ValueError("Must assign proper loss function to child.loss.")
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader)
            self._log.info("Epoch {}: Loss: {}".format(epoch, loss))

    def train_epoch(self, train_loader):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch

            #inputs, labels = inputs.to(self.device), labels.to(self.device)

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

    def hard_classification(self, inputs):
        """Hard classification from forwards' probability distribution
        """

        predicted_distribution = self.forward(inputs)
        class_ind, confidence = utils.tensor_argmax(predicted_distribution)
        return class_ind, confidence


class Ensemble():
    def __init__(self):
        self.members = list()
        self._log = logging.getLogger(self.__class__.__name__)

    def add_member(self, new_member):
        if issubclass(type(new_member), EnsembleMember):
            self._log.info("Adding {} to ensemble".format(type(new_member)))
            self.members.append(new_member)
        else:
            err_str = "Ensemble member must be an EnsembleMember subclass"
            self._log.error(err_str)
            raise ValueError(err_str)

    def add_multiple(self, number_of, constructor):
        for _ in range(number_of):
            self.add_member(constructor())

    def train(self, train_loader, num_epochs):
        """Multithreaded?"""
        self._log.info("Training ensemble")
        for member in self.members:
            member.train(train_loader, num_epochs)

    def predict(self, x, t=1):
        pred = list()
        for member in self.members:
            pred.append(member.predict(x, t))  # For future use rather

        pred_mean = torch.zeros([x.shape[0], self.members[0].output_size],
                                dtype=torch.float32)
        for p in pred:
            pred_mean += (1 / len(self.members)) * p

        return pred_mean
