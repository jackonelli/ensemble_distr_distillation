import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
import numpy as np
import os


class DummyLogitsProbabilityDistribution(distilled_network.DistilledNet):
    """We do "dummy" distillation and make the output independent of the input"""
    def __init__(self,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 scale_teacher_logits=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.gaussian_neg_log_likelihood,
            device=device)

        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate
        self.scale_teacher_logits = scale_teacher_logits

        #self.features = features
        self.par = nn.Parameter(torch.zeros((1, 18)))

        self.optimizer = torch_optim.SGD(self.parameters(),
                                          lr=self.learning_rate,
                                          momentum=0.9)

        self.to(self.device)

    def forward(self, x):
        """Estimate parameters of distribution
        """

        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]

        # We make the output independent of the input
        #y = self.features(torch.ones((1, 1)))
        #x = y.repeat(x[0].shape[0], 1)

        x = self.par[:, :18].repeat(x.shape[0], 1)
        mid = int(x.shape[-1] / 2)
        mean = x[:, :mid]
        var = torch.log(1 + torch.exp(x[:, mid:])) + 0.001 #+ nn.ReLU()(self.par[:, 18:])

        return mean, var

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

        if self.scale_teacher_logits:
            scaled_logits = logits - torch.stack([logits[:, :, -1]], axis=-1)
            logits = scaled_logits[:, :, 0:-1]

        return logits

    def predict(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if isinstance(input_, list) or isinstance(input_, tuple):
            input_ = input_[0]

        if num_samples is None:
            num_samples = len(self.teacher.members)

        mean, var = self.forward(input_)

        samples = torch.zeros([input_.size(0), num_samples, int(mean.size(-1))])
        for i in range(input_.size(0)):
            rv = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean[i, :],
                                                                            covariance_matrix=torch.diag(var[i, :]))
            samples[i, :, :] = rv.rsample([num_samples])

        if self.scale_teacher_logits:
            samples = torch.cat((samples, torch.zeros(samples.size(0), num_samples, 1)), dim=-1)

        return nn.Softmax(dim=-1)(samples)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """

        return self.loss(outputs, teacher_predictions)

    def _learning_rate_condition(self, epoch=None):
        if (epoch%10) == 0 and epoch <= 50:
            return True
        else:
            return False
