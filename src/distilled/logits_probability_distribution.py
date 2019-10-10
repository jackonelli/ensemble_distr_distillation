import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network


class LogitsProbabilityDistribution(distilled_network.DistilledNet):
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.gaussian_neg_log_likelihood,
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

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)

        self.to(self.device)

    def forward(self, x):
        """Estimate parameters of distribution
        """

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        mean = x[:, :int((self.output_size / 2))]
        var = torch.exp(-x[:, int((self.output_size / 2)):])

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

        return scaled_logits[:, :, 0:-1]  # ???

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

        return torch.nn.Softmax(dim=-1)(samples)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs, teacher_predictions)
