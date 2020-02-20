import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network


# TODO: remove??
class VanillaDistill(distilled_network.DistilledNet):
    def __init__(self,
                 layer_sizes,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 use_teacher_hard_labels=False,
                 learning_rate=0.001,
                 temp=10):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.mse,
            device=device)
        print(self.device)

        self.use_hard_labels = use_hard_labels
        self.use_teacher_hard_labels = use_teacher_hard_labels
        self.learning_rate = learning_rate
        self.temp = temp

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

        return torch.exp(x/self.temp) / (torch.sum(torch.exp(x/self.temp), dim=-1, keepdim=True) + 1)

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
        scaled_logits = logits - torch.stack([logits[:, :, -1]], dim=-1)
        teacher_pred = torch.exp(scaled_logits / self.temp) / (torch.sum(torch.exp(scaled_logits / self.temp), dim=-1,
                                                                         keepdim=True))

        return teacher_pred[:, :, :-1]

    def predict(self, input_):
        """Predict parameters
        Wrapper function for the forward function.
        """

        temp = self.temp
        self.temp = 1  # A bit of an ugly hack atm
        x = self.forward(input_)
        self.temp = temp

        return x

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

    def mean_output(self, outputs, teacher_predictions):
        # Not sure if this makes sense, but want some kind of convergence proof for the parameters
        return torch.mean(outputs, dim=0)




