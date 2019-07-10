"""Ensemble class"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import logging
import utils
import metrics


class EnsembleMember(nn.Module, ABC):
    """Parent class for keeping common logic in one place"""

    def __init__(self, loss_function, device=torch.device("cpu")):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.loss = loss_function
        self.optimizer = None
        self._log.info("Moving model to device: {}".format(device))
        self.device = device

    def train(self, train_loader, num_epochs, metrics_dict):
        if not isinstance(metrics_dict, metrics.MetricsDict):
            self.log.error(
                "Metrics must be of type MetricsDict, got {}".format(
                    type(metrics_dict)))
        if self.loss is None or not issubclass(type(self.loss),
                                               nn.modules.loss._Loss):
            raise ValueError("Must assign proper loss function to child.loss.")
        for epoch in range(1, num_epochs + 1):
            self._train_epoch(train_loader, metrics_dict)
            print("Epoch: {} {}".format(epoch, metrics_dict))

    def _train_epoch(self, train_loader, metrics_dict=None):
        """Train single epoch"""
        for batch in train_loader:
            metrics_dict.reset()
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.forward(inputs)

            loss = self.calculate_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            metrics_dict.update(loss, outputs, labels)

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

    def hard_classification(self, inputs):
        predicted_distribution = self.predict(inputs)
        class_ind, confidence = utils.tensor_argmax(predicted_distribution)

        return class_ind, confidence

    def save_ensemble(self, filepath):

        members_dict = {}
        for i, member in enumerate(self.members):
            members_dict["ensemble_member_{}".format(
                i
            )] = member  # To save memory one should save model.state_dict, but then we also need to save class-type etc., so I will keep it like this for now

        torch.save(members_dict, filepath)

    def load_ensemble(self, filepath):

        check_point = torch.load(filepath)

        for key in check_point:
            member = check_point[key]
            # member.eval(), should be called if we have dropout or batch-norm in our layers, to make sure that self.train = False, just that it doesn't work for now
            self.add_member(member)


class DistilledNet(nn.Module, ABC):
    """Parent class for distilled net logic in one place"""

    def __init__(self, teacher, loss_function, device=torch.device("cpu")):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.teacher = teacher
        self.loss = loss_function
        self.optimizer = None
        self._log.info("Moving model to device: {}".format(device))
        self.device = device

    def train(self, train_loader, num_epochs):
        if self.loss is None or not issubclass(type(self.loss),
                                               nn.modules.loss._Loss):
            self._log.warning(
                "Must assign proper loss function to child.loss.")
        for epoch in range(1, num_epochs + 1):
            loss = self._train_epoch(train_loader)
            self._log.info("Epoch {}: Loss: {}".format(epoch, loss))

    def _train_epoch(self, train_loader):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

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
