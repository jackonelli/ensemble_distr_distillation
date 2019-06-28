import numpy as np
import torch
import torch.nn as nn
import torch.optim as torch_optim
import loss as custom_loss


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size,
                 teacher, lr=0.001):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.lr = lr

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]

        self.loss = nn.CrossEntropyLoss()
        self.teacher = teacher

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.lr,
                                         momentum=0.9)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def calculate_loss(self, inputs, labels, t):
        outputs = self.forward(inputs)
        soft_targets = self.teacher.predict(inputs, t)

        loss = custom_loss.CrossEntropyLossOneHot.apply(outputs, soft_targets)

        if labels is not None:
            loss += self.loss(outputs, labels.type(torch.LongTensor))

        return loss

    def train_epoch(self, train_loader, t=1, hard_targets=False):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch

            if hard_targets:
                loss = self.calculate_loss(inputs=inputs, t=t)
            else:
                loss = self.calculate_loss(inputs=inputs, labels=labels, t=t)

            loss.sum().backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    def train(self, train_loader, num_epochs):

        epoch_half = np.floor(num_epochs / 2).astype(np.int)

        for epoch in range(1, epoch_half):
            loss = self.train_epoch(train_loader)
            print("Epoch {}: Loss: {}".format(epoch, loss))

        for epoch in range(epoch_half, num_epochs + 1):
            loss = self.train_epoch(train_loader, hard_targets=True)
            print("Epoch {}: Loss: {}".format(epoch, loss))

    def predict(self, x, t=1):
        x = self.forward(x)
        x = self.temperature_softmax(x, t)

        return x


def main():
    net = NeuralNet(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
