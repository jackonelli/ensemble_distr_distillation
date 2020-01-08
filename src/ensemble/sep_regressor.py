import itertools
import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble
from src.ensemble import mean_regressor
import src.loss as custom_loss


class SepRegressor(ensemble.EnsembleMember):
    """SepRegressor
    Network that predicts the parameters of a normal distribution

    Args:
        layer_sizes (list(int)): Defines the (equal) subnetworks,
            i.e. the last element (output_size) is D
        device (torch.Device)
        learning_rate (float)
    """
    def __init__(self,
                 layer_sizes,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        # The actual output the output of the combined subnetworks
        super().__init__(output_size=layer_sizes[-1] * 2,
                         target_size=layer_sizes[-1],
                         loss_function=custom_loss.gaussian_neg_log_likelihood,
                         device=device)

        self.learning_rate = learning_rate
        self.mu_network = mean_regressor.MeanRegressor(
            layer_sizes=layer_sizes,
            device=device,
            learning_rate=self.learning_rate)

        self.sigma_sq_network = mean_regressor.MeanRegressor(
            layer_sizes=layer_sizes,
            device=device,
            learning_rate=self.learning_rate)

        self.optimizer = torch_optim.SGD(itertools.chain(
            self.mu_network.parameters(), self.sigma_sq_network.parameters()),
                                         lr=self.learning_rate,
                                         momentum=0.9)
        self.to(self.device)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        sigma_sq_logit = self.sigma_sq_network.forward(x)
        logits = torch.cat((mu, sigma_sq_logit), dim=1)

        return logits

    def transform_logits(self, logits):
        # mean = logits[:, :int((self.output_size / 2))]
        # var = torch.exp(logits[:, int((self.output_size / 2)):])

        outputs = logits
        outputs[:, 1] = torch.log(1 + torch.exp(outputs[:, 1]))

        return outputs

    def calculate_loss(self, outputs, targets):
        mean = outputs[:, 0].reshape((outputs.size(0), 1))
        var = outputs[:, 1].reshape((outputs.size(0), 1))
        parameters = (mean, var)
        return self.loss(parameters, targets)

    def predict(self, x):
        logits = self.forward(x)
        x = self.transform_logits(logits)

        return x

    def _output_to_metric_domain(self, outputs):
        """Transform output for metric calculation
        Output distribution parameters are not necessarily
        exact representation for metrics calculation.
        This helper function can be overloaded to massage the output
        into the correct shape

        Extracts mean value
        """
        return outputs[:, 0]
