import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble
import src.loss as custom_loss
import src.utils as utils


class SimpleRegressor(ensemble.EnsembleMember):
    """SimpleRegressor
    Network that predicts the parameters of a normal distribution
    """
    def __init__(self,
                 layer_sizes,
                 device=torch.device("cpu"),
                 learning_rate=0.001,
                 variance_transform=utils.variance_linear_asymptote):

        super().__init__(output_size=layer_sizes[-1] // 2,
                         loss_function=custom_loss.gaussian_neg_log_likelihood,
                         device=device)

        self.learning_rate = learning_rate
        self.mean_only = False
        self.variance_transform = variance_transform
        self._log.info("Using variance transform: {}".format(
            self.variance_transform.__name__))

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.Adam(self.parameters(),
                                          lr=self.learning_rate)
        self.to(self.device)

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)
        # Add normalise here?

        return x

    def transform_logits(self, logits):
        """Transform logits for the simple regressor

        Maps half of the "logits" from unbounded to positive real numbers.
        TODO: Only works for one-dim output.

        Args:
            logits (torch.Tensor(B, D)):
        """

        outputs = logits
        outputs[:, 1] = self.variance_transform(outputs[:, 1])

        return outputs

    def calculate_loss(self, outputs, targets):
        mean = outputs[:, 0].reshape((outputs.size(0), 1))
        var = outputs[:, 1].reshape((outputs.size(0), 1))
        parameters = (mean, var)
        loss = None
        if self.mean_only:
            loss_function = nn.MSELoss()
            loss = loss_function(mean, targets)
        else:
            loss = self.loss(parameters, targets)
        return loss

    def predict(self, x):
        logits = self.forward(x)
        x = self.transform_logits(logits)

        return x

    def _output_to_metric_domain(self, outputs):
        """Transform output for metric calculation

        Extract expected value parameter from outputs
        """
        B, D = outputs.shape
        return outputs[:, 0].reshape((B, D // 2))
