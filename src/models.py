import torch
import torch.optim as torch_optim
import torch.nn as nn
import loss as custom_loss
import ensemble


class NeuralNet(ensemble.EnsembleMember):
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

    def calculate_loss(self, inputs, labels):
        outputs = self.forward(inputs)

        return self.loss(outputs, labels.type(torch.LongTensor))

    def predict(self, x, t=1):
        x = self.forward(x)
        x = self.temperature_softmax(x, t)

        return x


class LinearNet(ensemble.EnsembleMember):
    """Simple linear binary classifier"""

    def __init__(self, input_size=2, output_size=2, learning_rate=0.01):
        super().__init__(loss_function=nn.CrossEntropyLoss())

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.learning_rate,
                                         momentum=0.9)

    def forward(self, x):
        x = self.linear(x)
        return x

    def calculate_loss(self, inputs, labels):
        output = self.forward(inputs)

        return self.loss(output, labels.type(torch.LongTensor))


def main():
    net = NeuralNet(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
