import logging
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as torch_optim
import loss as custom_loss
import utils


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
            # raise ValueError("Must assign proper loss function to child.loss.")
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
            _, teacher_predictions = self.teacher.predict(inputs)

            outputs = self.forward(inputs)
            loss = self.calculate_loss(outputs, teacher_predictions, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    def teacher_sufficient_statistics(self, teacher_predictions):
        """Base class implements default transform
        Default sufficient statistics
        is simply the unit transformation of the teacher predictions
        """
        return teacher_predictions

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        pass

    def hard_classification(self, inputs):
        """Hard classification from forwards' probability distribution
        """

        predicted_distribution = self.forward(inputs)
        class_ind, confidence = utils.tensor_argmax(predicted_distribution)
        return class_ind, confidence


class PlainProbabilityDistribution(DistilledNet):
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        # super().__init__(nn.NLLLoss(), device=device)
        super().__init__(
            loss_function=custom_loss.dirichlet_neg_log_likelihood,
            teacher=teacher,
            device=device)

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]

        self.teacher = teacher

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)
        self.to(self.device)

    def forward(self, x, t=1):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.softmax(x / t, dim=-1)

        return x

    def predict(self, x):  # For convenience
        return self.forward(x)

    def calculate_loss(self, inputs, labels, t):
        outputs = self.forward(inputs)
        soft_targets = self.teacher.predict(inputs, t)

        loss = custom_loss.scalar_loss(outputs, soft_targets)

        if labels is not None and self.use_hard_labels:
            loss += self.loss(outputs, labels.type(torch.LongTensor))

        return loss

    def train_epoch(self, train_loader, t):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            loss = self.calculate_loss(inputs=inputs, labels=labels, t=t)

            loss.sum().backward()

            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    def train(self, train_loader, num_epochs, t=1):

        #epoch_half = np.floor(num_epochs / 2).astype(np.int)
        self._log.info("Training distilled network.")
        self.use_hard_labels = True
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader, t=t)
            self._log.info("Epoch {}: Loss: {}".format(epoch, loss))

            #if epoch == (epoch_half + 1):
            #  self.use_hard_labels = True


class DirichletProbabilityDistribution(DistilledNet):
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.dirichlet_neg_log_likelihood,
            device=device)

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)

        self.to(self.device)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = x + 1

        return x

    # OBS: SOM DET ÄR UPPBYGGT NU, FUNGERAR INTE HARD-CLASSIFICATION I DISTILLED_NET-PARENT-KLASSEN
    def predict(self, x):
        #  alphas = self.forward(x)
        #  strength = torch.sum(alphas, dim=-1).unsqueeze(dim=1)
        #  p_hat = torch.div(alphas, strength)

        #  return p_hat
        return self.forward(x)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        return self.loss(outputs, teacher_predictions)

    #  def calculate_loss(self, outputs, suff_stats, labels=None):
    #      soft_targets = self.teacher.predict(inputs, t)

    #      if labels is not None and self.use_hard_labels:
    #          lambda_t = np.min([1.0, t / 10])
    #          hard_targets = utils.to_one_hot(labels, self.output_size).type(
    #              torch.FloatTensor)
    #          loss = custom_loss.sum_of_squares_bayes_risk(
    #              alphas, soft_targets, hard_targets, lambda_t)
    #      else:
    #          loss = custom_loss.sum_of_squares_bayes_risk(alphas, soft_targets)

    #      return loss

    def train(self, train_loader, num_epochs, t=1):

        #  scheduler = torch_optim.lr_scheduler.StepLR(self.optimizer,
        #                                               step_size=5,
        #                                               gamma=0.1)
        self.use_hard_labels = True

        self._log.info("Training distilled network.")
        for epoch in range(1, num_epochs + 1):
            loss = self._train_epoch(train_loader)
            self._log.info("Epoch {}: Loss: {}".format(epoch, loss))


def main():
    net = PlainProbabilityDistribution(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
