import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network


class NiwProbabilityDistribution(distilled_network.DistilledNet):
    def __init__(self,
                 layer_sizes,
                 target_dim,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.gaussian_inv_wishart_neg_log_likelihood,
            device=device)

        self.target_dim = target_dim
        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)

        self.to(self.device)

    def forward(self, x):
        """Estimate distribution parameters
        """

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        mu = x[:, :self.target_dim]
        scale = torch.exp(x[:, self.target_dim:(self.target_dim+1)])
        psi = torch.exp(x[:, (self.target_dim+1):(2*self.target_dim+1)])
        # Degrees of freedom should be larger than D - 1
        # Temporary way of defining that
        nu = torch.exp(x[:, (2*self.target_dim+1):]) + (self.target_dim - 1)

        return mu, scale, psi, nu

    def predict(self, input_):
        """Predict parameters
        Wrapper function for the forward function.
        """
        return torch.cat(self.forward(input_), dim=-1)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs, (teacher_predictions[:, :, :self.target_dim],
                                   teacher_predictions[:, :, self.target_dim:]))
