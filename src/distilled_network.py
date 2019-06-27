import torch
import torch.nn as nn
import cross_entropy_loss_one_hot


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, teacher):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]

        self.teacher = teacher

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def loss(self, x, target):
        output = self.forward(x)
        loss = nn.CrossEntropyLoss()

        return loss(output, target)

    def loss_cross_entropy_soft_targets(self, x, t=1):
        output = self.forward(x)
        target = self.teacher.prediction(x, t)

        loss = cross_entropy_loss_one_hot.CrossEntropyLossOneHot.apply(output, target)

        return loss

    def predict(self, x, t=1):
        x = self.forward(x)
        x = self.temperature_softmax(x, t)

        return x


def main():
    net = NeuralNet(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
