import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network


class DummyLogitsProbabilityDistribution(distilled_network.DistilledNet):
    """We do "dummy" distillation and make the output independent of the input"""
    def __init__(self,
                 layer_sizes,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.gaussian_neg_log_likelihood,
            device=device)

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
        """Estimate parameters of distribution
        """

        # We make the output independent of the input
        x = torch.ones(x.size())
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        mid = int(x.shape[-1] / 2)
        mean = x[:, :mid]
        var = torch.exp(x[:, mid:])

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

        scaled_logits = logits - torch.stack([logits[:, :, -1]], axis=-1)

        return scaled_logits[:, :, 0:-1]

    def predict(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        if num_samples is None:
            num_samples = len(self.teacher.members)

        mean, var = self.forward(input_)

        samples = torch.zeros([input_.size(0), num_samples, int(self.output_size / 2)])
        for i in range(input_.size(0)):
            rv = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean[i, :],
                                                                            covariance_matrix=torch.diag(var[i, :]))
            samples[i, :, :] = rv.rsample([num_samples])

        softmax_samples = torch.exp(samples) / (torch.sum(torch.exp(samples), dim=-1, keepdim=True) + 1)

        return softmax_samples

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs, teacher_predictions)
