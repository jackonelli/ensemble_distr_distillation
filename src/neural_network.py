import torch.nn as nn


class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.layers = self.initialize_network()

    def initialize_network(self):
        layers = []

        fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        layers.append(fc1)
        layers.append(fc2)
        layers.append(fc3)

        return layers

    def forward(self, x):

        assert x.size[1] == self.input_size

        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x))

            if i == (len(self.layers) - 1):
                x = nn.functional.softmax(x)

        return x
