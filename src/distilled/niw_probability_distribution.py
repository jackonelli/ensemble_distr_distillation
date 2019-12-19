"""Distilled Normal Inverse Wishart model

Distills an ensemble where each member predicts the actual mean and variance
"""
import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network


class NiwProbabilityDistribution(distilled_network.DistilledNet):
    """NiwProbabilityDistribution
    Sub class of DistilledNet
    """
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 target_dim,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.gaussian_inv_wishart_neg_log_likelihood,
            device=device)

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.target_dim = target_dim
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
        """Estimate distribution parameters
        """

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        #print("x", x[:, self.target_dim:(self.target_dim + 1)])

        mu_0 = x[:, :self.target_dim]
        lambda_ = torch.log(1.0 +
                            torch.exp(x[:, self.target_dim:(self.target_dim +
                                                            1)]))
        #print("lambda", lambda_)
        psi = torch.exp(x[:, (self.target_dim + 1):(2 * self.target_dim + 1)])

        # Degrees of freedom should be larger than D - 1
        nu = torch.exp(
            x[:, (2 * self.target_dim + 1):]) + (self.target_dim - 1)

        return mu_0, lambda_, psi, nu

    def predict(self, input_):
        """Predict parameters
        Wrapper function for the forward function.
        """
        return torch.cat(self.forward(input_), dim=-1)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """

        return self.loss(outputs,
                         (teacher_predictions[:, :, :self.target_dim],
                          teacher_predictions[:, :, self.target_dim:]))
