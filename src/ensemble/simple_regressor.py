import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble
import src.loss as custom_loss


class SimpleRegressor(ensemble.EnsembleMember):
    """Regression network that predicts the parameters of a normal distribution"""
    # ELLER KAN KANSKE ANVÄDA DENNA FÖR NIW OCKSÅ, MEN FÅR FIXA HUR JAG HANTERAR OUTPUTEN

    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        super().__init__(loss_function=custom_loss.gaussian_neg_log_likelihood, device=device)

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
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
        x = self.fc3(x)

        mean = x[:, :(self.output_size / 2)]
        var = torch.log(x[:, (self.output_size / 2):])

        return mean, var

    def calculate_loss(self, outputs, targets):
        return self.loss(outputs, targets)

    def predict(self, x):
        x = self.forward(x)

        return x
