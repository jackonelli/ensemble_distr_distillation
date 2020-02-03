import torch
import torch.nn as nn
import torch.optim as torch_optim
import src.loss as custom_loss
import src.distilled.distilled_network as distilled_network

# TODO: what's the difference between this class and the one in dirichlet_probability_distribution
class DirichletCNN(distilled_network.DistilledNet):
    def __init__(self,
                 layer_sizes,
                 teacher,
                 device=torch.device('cpu'),
                 use_hard_labels=False,
                 learning_rate=0.001):
        super().__init__(
            teacher=teacher,
            loss_function=custom_loss.dirichlet_neg_log_likelihood,
            device=device)

        self.use_hard_labels = use_hard_labels
        self.learning_rate = learning_rate

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)

        self.to(self.device)

    def forward(self, x):
        """Estimate alpha parameters
        Note the + 1 shift.
        This was added for stability (Dirichlet support: alpha > 0)
        not ideal since we might want alpha < 1
        """

        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))

        x = self.layers[-1](x)
        x = x + 1

        return x

    def predict(self, input_):
        """Predict parameters
        Wrapper function for the forward function.
        """
        return self.forward(input_)

    def calculate_loss(self, outputs, teacher_predictions, labels=None):
        """Calculate loss function
        Wrapper function for the loss function.
        """
        return self.loss(outputs, teacher_predictions)
