import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble
import src.loss as custom_loss


class SimpleRegressor(ensemble.EnsembleMember):
    """SimpleRegressor
    Network that predicts the parameters of a normal distribution
    """
    def __init__(self,
                 layer_sizes,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        super().__init__(
            loss_function=custom_loss.gaussian_neg_log_likelihood_1d,
            device=device)

        self.learning_rate = learning_rate

        # Ad-hoc fix zero variance.
        self.variance_lower_bound = 0.0
        if self.variance_lower_bound > 0.0:
            self._log.warning("Non-zero variance lower bound set ({})".format(
                self.variance_lower_bound))

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)
        self.to(self.device)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)
        # Add normalise here?

        return x

    def transform_logits(self, logits):
        # mean = logits[:, :int((self.output_size / 2))]
        # var = torch.exp(logits[:, int((self.output_size / 2)):])

        outputs = logits
        outputs[:, 1] = torch.log(
            1 + torch.exp(outputs[:, 1])) + self.variance_lower_bound

        return outputs

    def calculate_loss(self, outputs, targets):
        return self.loss(outputs, targets)

    def predict(self, x):
        logits = self.forward(x)
        x = self.transform_logits(logits)

        return x
