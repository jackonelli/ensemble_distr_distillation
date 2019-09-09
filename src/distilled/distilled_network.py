import logging
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as torch_optim


class DistilledNet(nn.Module, ABC):
    """Parent class for distilled net logic in one place"""

    def __init__(self, teacher, loss_function, device=torch.device("cpu")):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.teacher = teacher
        self.loss = loss_function
        self.metrics = dict()
        if self.loss is None or not issubclass(type(self.loss),
                                               nn.modules.loss._Loss):
            # raise ValueError("Must assign proper loss function to child.loss.")
            self._log.warning(
                "Must assign proper loss function to child.loss.")
        self.optimizer = None
        self._log.info("Moving model to device: {}".format(device))
        self.device = device

    def train(self, train_loader, num_epochs):
        """ Common train method for all distilled networks
        Should NOT be overridden!
        """

        scheduler = torch_optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=5,
                                                    gamma=0.1)
        self.use_hard_labels = False

        self._log.info("Training distilled network.")
        for epoch_number in range(1, num_epochs + 1):
            loss = self._train_epoch(train_loader)
            self._print_epoch(epoch_number, loss)
            if self._learning_rate_condition(epoch_number):
                scheduler.step()

    def _train_epoch(self, train_loader):
        """Common train epoch method for all distilled networks
        Should NOT be overridden!
        TODO: Make sure train_loader returns None for labels,
        if no labels are available.
        """
        running_loss = 0
        self._reset_metrics()
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            teacher_predictions = self.teacher.predict(inputs, t=None)  # TO DO: FIX THE t SO THAT THIS FUNCTION IS MORE FLEXIBLE

            outputs = self.forward(inputs)

            loss = self.calculate_loss(outputs, teacher_predictions, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            self._update_metrics(outputs, labels)
        return running_loss

    def _add_metric(self, metric):
        self.metrics[metric.name] = metric

    def _update_metrics(self, outputs, labels):
        for metric in self.metrics.values():
            metric.update(labels=labels, outputs=outputs)

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _print_epoch(self, epoch_number, loss):
        epoch_string = "Epoch {}: Loss: {}".format(epoch_number, loss)
        for metric in self.metrics.values():
            epoch_string += " {}".format(metric)
        self._log.info(epoch_string)

    def _learning_rate_condition(self, epoch):
        """Evaluate condition for increasing learning rate
        Defaults to never increasing. I.e. returns False
        """

        return False

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        pass
