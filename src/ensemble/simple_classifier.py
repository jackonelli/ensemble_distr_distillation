import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble


class SimpleClassifier(ensemble.EnsembleMember):
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 device=torch.device("cpu"),
                 learning_rate=0.001):

        super().__init__(loss_function=nn.NLLLoss(), device=device)
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
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
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def transform_logits(self, logits):
        """Should this be log softmax?""" # Men vi har lagt det i calculate_loss istället
        return (nn.Softmax(dim=-1))(logits)

    def calculate_loss(self, outputs, labels):
        log_outputs = torch.log(outputs)
        # Removing this since it gives nan loss

        return self.loss(log_outputs, labels.type(torch.LongTensor))

    def predict(self, x, t=1):
        x = self.forward(x)
        x = self.transform_logits(x)

        return x
