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
        var = torch.exp(x[:, int((self.output_size / 2)):])

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

        return self.teacher.get_logits(inputs)

    def predict(self, input_):
        """Predict parameters
        Wrapper function for the forward function.
        """
        return torch.cat(self.forward(input_), dim=-1)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs, teacher_predictions)
