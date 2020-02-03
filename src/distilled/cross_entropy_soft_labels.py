import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network
from src import utils

# TODO: Remove? Was just a test and not really relevant anymore
class XCSoftLabels(distilled_network.DistilledNet):
    """We match only the mean of the logits"""

    def __init__(self,
                 layer_sizes,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 use_teacher_hard_labels=True,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.cross_entropy_soft_targets,
            device=device)
        print(self.device)

        self.use_hard_labels = use_hard_labels
        self.use_teacher_hard_labels = use_teacher_hard_labels
        self.learning_rate = learning_rate

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.Adam(self.parameters(),
                                          lr=self.learning_rate)

        self.to(self.device)

    def forward(self, x):

        """Estimate parameters of distribution
        """

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)

        x = torch.exp(x) / (torch.sum(torch.exp(x), dim=-1, keepdim=True) + 1)

        return x

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
        scaled_logits = logits - torch.cat([logits[:, :, -1]], axis=-1)
        teacher_pred = self.teacher.transform_logits(scaled_logits)

        if self.use_teacher_hard_labels:
            teacher_pred = torch.argmax(teacher_pred, dim=-1)
            teacher_pred = utils.to_one_hot(teacher_pred, number_of_classes=self.output_size + 1)\
                .type(torch.FloatTensor)

        return teacher_pred

    def predict(self, input_, num_samples=None):
        """Predict parameters
        Wrapper function for the forward function.
        """

        return self.forward(input_)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """

        return self.loss(outputs, teacher_predictions[:, 0, :])

    def _learning_rate_condition(self, epoch=None):
        """Evaluate condition for increasing learning rate
        Defaults to never increasing. I.e. returns False
        """

        return True

    def logits_rmse(self, outputs, teacher_predictions):
        # This will not be the nicest solution, but well well
        last_prob_target = 1 - torch.sum(teacher_predictions, dim=1, keepdim=True)
        target_total = 1 / last_prob_target
        target_logits = torch.log(teacher_predictions * target_total)

        last_prob_predictions = 1 - torch.sum(outputs, dim=1, keepdim=True)
        prediction_total = 1 / last_prob_predictions
        prediction_logits = torch.log(outputs * prediction_total)

        return custom_loss.mse(prediction_logits, target_logits)

