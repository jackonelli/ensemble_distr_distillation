import torch
import torch.optim as torch_optim
import torch.nn as nn
from src.ensemble import ensemble


class ConvolutionalClassifier(ensemble.EnsembleMember):
    def __init__(self, device=torch.device("cpu"), learning_rate=0.001):
        super().__init__(loss_function=nn.CrossEntropyLoss(), device=device)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=learning_rate,
                                         momentum=0.9)
        self.to(self.device)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def calculate_loss(self, outputs, labels):
        return self.loss(outputs, labels)


class AbaConvolutionalClassifier(ensemble.EnsembleMember):
    """Obsolete"""

    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 device=torch.device("cpu"),
                 learning_rate=0.001):
        super().__init__(loss_function=nn.CrossEntropyLoss(), device=device)
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

    @staticmethod
    def temperature_softmax(x, t=1):
        return nn.functional.softmax(x / t, dim=-1)

    def calculate_loss(self, outputs, labels):

        return self.loss(outputs, labels.type(torch.LongTensor))

    def predict(self, x, t=1):
        x = self.forward(x)
        x = self.temperature_softmax(x, t)

        return x
