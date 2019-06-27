import torch.nn as nn


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x, t=1):

        assert x.size[1] == self.input_size

        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x))

            if i == (len(self.layers) - 1):
                x = self.temperature_softmax(x, t)
        return x

    @staticmethod
    def temperature_softmax(x, t=1):
        return nn.functional.softmax(x / t)

    def loss(self, x, target):
        output = self.forward(x)
        loss = nn.CrossEntropyLoss()

        return loss(output, target)


def main():
    net = NeuralNet(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
