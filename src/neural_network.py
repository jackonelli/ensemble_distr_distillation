import torch
import torch.nn as nn
import torch.optim as torch_optim
import cross_entropy_loss_one_hot


class DumpNet(nn.Module):
    """Sanity check CIFAR model from here:
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""

    def __init__(self, lr=0.001):
        super(DumpNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.lr,
                                         momentum=0.9)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_full(self, train_loader):
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.calculate_loss(inputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def calculate_loss(self, inputs, target):
        output = self.forward(inputs)
        return self.loss(output, target)


class LinearNet(nn.Module):
    """Simple linear binary classifier"""

    def __init__(self, input_size=2, output_size=2, lr=0.01):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.lr,
                                         momentum=0.9)

    def forward(self, x, t=1):

        x = self.linear(x)
        return x

    def calculate_loss(self, x, target):
        output = self.forward(x)
        loss = nn.CrossEntropyLoss()

        return loss(output, target)

    def train_epoch(self, train_loader):
        """Train single epoch"""
        running_loss = 0
        # print("parameters before", list(self.parameters())[0])
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            loss = self.calculate_loss(inputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            #print_batch(batch, loss)
        return running_loss


def print_batch(batch, loss):
    inputs, labels = batch
    for ind, input_ in enumerate(inputs):
        print("{}: {}\nloss: {}".format(input_.data.numpy(), labels[ind],
                                        loss))


class NeuralNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 lr=0.001):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.lr = lr

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]
        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.lr,
                                         momentum=0.9)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        # for i, layer in enumerate(self.layers):
        #     x = nn.functional.relu(layer(x))

        #     if i == (len(self.layers) - 1):
        #         x = self.temperature_softmax(x, t)
        # return x
        return x

    @staticmethod
    def temperature_softmax(x, t=1):
        return nn.functional.softmax(x / t, dim=-1)

    def train_epoch(self, train_loader):
        """Train single epoch"""
        running_loss = 0
        #print("parameters before", list(self.parameters())[0])
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            loss = self.loss(inputs, labels)
            loss.backward()
            # for p in self.parameters():
            #     if p.grad is not None:
            #         print("grad", p.grad.data)
            self.optimizer.step()
            running_loss += loss.item()
        #print("parameters after", list(self.parameters())[0])
        return running_loss

    def loss(self, x, target):
        output = self.forward(x)
        loss = nn.CrossEntropyLoss()

        return loss(output, target.type(torch.LongTensor))

    def predict(self, x, t=1):
        x = self.forward(x)
        x = self.temperature_softmax(x, t)

        return x


def main():
    net = NeuralNet(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
