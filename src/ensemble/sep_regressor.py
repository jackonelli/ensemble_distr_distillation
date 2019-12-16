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
    """
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        super().__init__(
            loss_function=custom_loss.gaussian_neg_log_likelihood_1d,
            device=device)

        self.learning_rate = learning_rate
        self.mu_network = mean_regressor.MeanRegressor(
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size // 2,
            device=device,
            learning_rate=self.learning_rate)

        self._log.debug(output_size // 2)
        self.sigma_sq_network = mean_regressor.MeanRegressor(
            input_size,
            hidden_size_1,
            hidden_size_2,
            output_size // 2,
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
        self._log.debug("Forward")
        self._log.debug(sigma_sq_logit.shape)

        return logits

    def transform_logits(self, logits):
        # mean = logits[:, :int((self.output_size / 2))]
        # var = torch.exp(logits[:, int((self.output_size / 2)):])

        self._log.debug("Transform logits")
        self._log.debug(logits.shape)
        outputs = logits
        outputs[:, 1] = torch.log(1 + torch.exp(outputs[:, 1]))

        return outputs

    def calculate_loss(self, outputs, targets):
        return self.loss(outputs, targets)

    def predict(self, x):
        logits = self.forward(x)
        x = self.transform_logits(logits)

        return x