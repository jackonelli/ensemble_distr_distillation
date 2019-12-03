"""Distilled net base module"""
import logging
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as torch_optim
import math
import src.utils as utils


class DistilledNet(nn.Module, ABC):
    """Parent class for distilled net logic in one place"""
    def __init__(self, teacher, loss_function, device=torch.device("cpu")):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.teacher = teacher
        self.loss = loss_function
        self.metrics = dict()
        self.use_hard_labels = False

        if self.loss is None or not issubclass(type(self.loss),
                                               nn.modules.loss._Loss):
            # raise ValueError(
            #    "Must assign proper loss function to child.loss.")
            self._log.warning(
                "Must assign proper loss function to child.loss.")
        self.optimizer = None
        self._log.info("Moving model to device: {}".format(device))
        self.device = device

    def train(self, train_loader, num_epochs, validation_loader=None):
        """ Common train method for all distilled networks
        Should NOT be overridden!
        """
        scheduler = self.get_scheduler(step_size=10 * len(train_loader),
                                       cyclical=True)

        # scheduler = torch_optim.lr_scheduler.CyclicLR(
        #    self.optimizer, 1e-7, 0.1, step_size_up=100)

        self._log.info("Training distilled network.")
        for epoch_number in range(1, num_epochs + 1):
            loss = self._train_epoch(train_loader,
                                     validation_loader=validation_loader,
                                     scheduler=scheduler)
            self._print_epoch(epoch_number, loss)
            if self._learning_rate_condition(epoch_number):
                scheduler.step()

    def _train_epoch(self,
                     train_loader,
                     validation_loader=None,
                     scheduler=None):
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
            teacher_predictions = self._generate_teacher_predictions(
                inputs).detach()

            outputs = self.forward(inputs)

            loss = self.calculate_loss(outputs, teacher_predictions, None)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if math.isnan(running_loss):
                break

            if validation_loader is None:
                #self._reset_metrics()
                self._update_metrics(
                    outputs, teacher_predictions
                )  # BUT THIS DOES NOT WORK FOR EG ACCURACY
                # USE EITHER METRICS.ACCURACY_SOFT_LABELS OR METRICS.ACCURACY_LOGITS

            if self._learning_rate_condition():
                scheduler.step()

        if validation_loader is not None:
            # We will compare here with the teacher predictions
            for valid_batch in validation_loader:
                #self._reset_metrics()
                valid_inputs, valid_labels = valid_batch
                valid_inputs, valid_labels = valid_inputs.to(
                    self.device), valid_labels.to(self.device)
                valid_outputs = self.forward(valid_inputs)
                teacher_predictions = self._generate_teacher_predictions(
                    valid_inputs)
                teacher_predictions = teacher_predictions.to(self.device)
                self._update_metrics(valid_outputs, teacher_predictions)

        return running_loss

    def _generate_teacher_predictions(self, inputs):
        """Generate teacher predictions
        The intention is to get the logits of the ensemble members
        and then apply some transformation to get the desired predictions.
        Default implementation is to recreate the exact ensemble member output.
        Override this method if another logit transformation is desired,
        e.g. unit transformation if desired predictions
        are the logits themselves
        """

        logits = self.teacher.get_logits(inputs)
        return self.teacher.transform_logits(logits)

    def calc_metrics(self, data_loader):
        self._reset_metrics()

        for batch in data_loader:
            inputs, targets = batch
            outputs = self.forward(inputs)
            teacher_predictions = self._generate_teacher_predictions(inputs)
            self._update_metrics(outputs, teacher_predictions)

        metric_string = ""
        for metric in self.metrics.values():
            metric_string += " {}".format(metric)
        self._log.info(metric_string)

    def get_scheduler(self, step_size, factor=100000, cyclical=False):

        if cyclical:
            end_lr = self.learning_rate
            clr = utils.cyclical_lr(step_size,
                                    min_lr=end_lr / factor,
                                    max_lr=end_lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, [clr])
        else:
            scheduler = torch_optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=100,
                                                        gamma=0.5)

        return scheduler

    def add_metric(self, metric):
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

    def _learning_rate_condition(self, epoch=None):
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
