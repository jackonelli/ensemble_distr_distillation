import torch
import torch.nn as nn
import torch.optim as torch_optim
import loss as custom_loss
import distilled.distilled_network as distilled_network


class DirichletCNN(distilled_network.DistilledNet):
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
            loss_function=custom_loss.dirichlet_neg_log_likelihood,
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
        """Estimate alpha parameters
        Note the + 1 shift.
        This was added for stability (Dirichlet support: alpha > 0)
        not ideal since we might want alpha < 1
        """

        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
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
