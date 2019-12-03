import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network


class LogitsMatching(distilled_network.DistilledNet):
    """We match only the mean of the logits"""
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 use_teacher_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.rmse,
            device=device)
        print(self.device)
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.use_hard_labels = use_hard_labels
        self.use_teacher_hard_labels = use_teacher_hard_labels
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]

        self.optimizer = torch_optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

        self.to(self.device)

    def forward(self, x):

        """Estimate parameters of distribution
        """

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        mean = x
        # Note: we actually don't learn the variance
        #var = torch.ones(mean.size())

        return mean#, var

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

        if self.use_teacher_hard_labels:
            pass
        else:
            scaled_logits = logits - torch.stack([logits[:, :, -1]], axis=-1)
            teacher_pred = scaled_logits[:, :, 0:-1]

        return teacher_pred

    def predict(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        mean = self.forward(input_)

        return torch.exp(mean) / (torch.sum(torch.exp(mean), dim=1, keepdim=True) + 1)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs, teacher_predictions)

    def _learning_rate_condition(self, epoch=None):
        """Evaluate condition for increasing learning rate
        Defaults to never increasing. I.e. returns False
        """

        return True

    def softmax_rmse(self, outputs, teacher_predictions):
        # We will convert this to the softmax output so that we can calculate metrics on them
        teacher_distribution = torch.exp(teacher_predictions) / (torch.sum(torch.exp(teacher_predictions), dim=-1,
                                                                           keepdim=True) + 1)
        predicted_distribution = torch.exp(outputs) / (torch.sum(torch.exp(outputs), dim=-1, keepdim=True) + 1)

        return custom_loss.rmse(predicted_distribution, teacher_distribution)

    def softmax_xentropy(self, outputs, teacher_predictions):
        # We will convert this to the softmax output so that we can calculate metrics on them
        # NOTE: this function is only defined for one ensemble member
        teacher_distribution = torch.exp(teacher_predictions[:, 0, :]) / (torch.sum(torch.exp(teacher_predictions[:, 0, :]),
                                                                                    dim=-1, keepdim=True) + 1)
        predicted_distribution = torch.exp(outputs) / (torch.sum(torch.exp(outputs), dim=-1, keepdim=True) + 1)

        return custom_loss.cross_entropy_soft_targets(predicted_distribution, teacher_distribution)
