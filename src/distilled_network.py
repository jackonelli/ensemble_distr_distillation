import numpy as np
import torch
import torch.nn as nn
import torch.optim as torch_optim
import loss as custom_loss
import ensemble
import utils

# Can we put the distilled net parent class here instead?


class PlainProbabilityDistribution(
        ensemble.EnsembleMember
):  # Should instead use the DistilledNet parent class? I think maybe we nee some reconstruction here
    """Not necessarily an ensemble member but can be used as one"""

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

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = torch.exp(self.fc3(x))

        return x

    def predict(self, x):  # For convenience
        return self.forward(x)

    def calculate_loss(self, outputs, labels, soft_targets, t):

        loss = self.loss(outputs, soft_targets)

        if labels is not None and self.use_hard_labels:
            loss += self.loss(outputs, labels)

        return loss

    def train_epoch(self, train_loader, metrics_dict, t):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.forward(inputs)
            soft_targets = self.teacher.predict(inputs, t)
            loss = self.calculate_loss(outputs=outputs,
                                       labels=labels,
                                       soft_targets=soft_targets,
                                       t=t)

            loss.backward()
            # print(loss)
            # print(outputs)
            # break

            self.optimizer.step()
            metrics_dict.update(loss, labels, outputs)
        return running_loss

    def train(self, train_loader, num_epochs, metrics_dict, t=1):

        #epoch_half = np.floor(num_epochs / 2).astype(np.int)
        self._log.info("Training distilled network.")
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader, metrics_dict, t=t)
            self._log.info("Epoch: {} {}".format(epoch, metrics_dict))

            #if epoch == (epoch_half + 1):
            #  self.use_hard_labels = True


class DirichletProbabilityDistribution(
        ensemble.EnsembleMember
):  # Should instead use the DistilledNet parent class?
    """Not necessarily an ensemble member but can be used as one"""

    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(nn.MSELoss(), device=device)

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

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))

        return x

    # OBS: SOM DET ÄR UPPBYGGT NU, FUNGERAR INTE HARD-CLASSIFICATION I DISTILLED_NET PARENT-KLASSEN
    def predict(self, x):
        alphas = self.forward(x) + 1
        strength = torch.sum(alphas, dim=-1).unsqueeze(dim=1)
        p_hat = torch.div(alphas, strength)

        return p_hat

    def calculate_loss(self, inputs, labels, t):
        alphas = self.forward(inputs) + 1
        soft_targets = self.teacher.predict(inputs, t)

        lambda_t = np.min([1.0, t / 10])
        loss = custom_loss.sum_of_squares_bayes_risk(alphas, soft_targets,
                                                     lambda_t)

        if labels is not None and self.use_hard_labels:
            loss = custom_loss.sum_of_squares_bayes_risk(
                alphas, soft_targets, lambda_t,
                utils.to_one_hot(labels,
                                 self.output_size).type(torch.FloatTensor))

            #strength = torch.sum(alphas, dim=-1).unsqueeze(dim=1)
            #p_hat = torch.div(alphas, strength)
            #loss += self.loss(p_hat, utils.to_one_hot(labels, self.output_size).type(torch.FloatTensor))

        return loss

    def train_epoch(self, train_loader, t):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            loss = self.calculate_loss(inputs=inputs, labels=labels, t=t)
            print(loss)

            loss.backward()

            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    def train(self, train_loader, num_epochs, t=1):

        #epoch_half = np.floor(num_epochs / 2).astype(np.int)

        self.use_hard_labels = True
        self._log.info("Training distilled network.")

        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader, t=t)
            self._log.info("Epoch {}: Loss: {}".format(epoch, loss))

            #if epoch == (epoch_half + 1):
            #  self.use_hard_labels = True


def main():
    net = PlainProbabilityDistribution(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
