import numpy as np
import torch
import torch.nn as nn
import torch.optim as torch_optim
import loss as custom_loss
import ensemble


class PlainProbabilityDistribution(ensemble.EnsembleMember):
    """Not necessarily an ensemble member but can be used as one"""

    def __init__(self,
                 input_size,
                 hidden_size_1,
                 hidden_size_2,
                 output_size,
                 teacher,
                 use_hard_labels=False,
                 lr=0.001):
        super().__init__(nn.NLLLoss())

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1  # Or make a list or something
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.use_hard_labels = use_hard_labels
        self.lr = lr

        self.fc1 = nn.Linear(self.input_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.output_size)

        self.layers = [self.fc1, self.fc2, self.fc3]

        self.teacher = teacher

        self.optimizer = torch_optim.SGD(self.parameters(),
                                         lr=self.lr,
                                         momentum=0.9)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=-1)

        return x

    def calculate_loss(self, inputs, labels, t):
        outputs = self.forward(inputs)
        soft_targets = self.teacher.predict(inputs, t)

        # Extra none loss
        # loss = custom_loss.CrossEntropyLossOneHot.apply(outputs, soft_targets)
        loss = custom_loss.scalar_loss(outputs, soft_targets)

        if labels is not None and self.use_hard_labels:
            loss += self.loss(outputs, labels.type(torch.LongTensor))

        return loss

    def train_epoch(self, train_loader, t, hard_targets=False):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch

            loss = self.calculate_loss(inputs=inputs, labels=labels, t=t)

            loss.sum().backward()

            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    def train(self, train_loader, num_epochs, t=1):

        epoch_half = np.floor(num_epochs / 2).astype(np.int)
        self._log.info("Training distilled network.")

        for epoch in range(1, epoch_half):
            loss = self.train_epoch(train_loader, t=t)
            self._log.info("Epoch {}: Loss: {}".format(epoch, loss))

        for epoch in range(epoch_half, num_epochs + 1):
            loss = self.train_epoch(train_loader, t=t, hard_targets=True)
            self._log.info("Epoch {}: Loss: {}".format(epoch, loss))


def main():
    net = PlainProbabilityDistribution(20, 10, 5, 2)
    print(net)


if __name__ == "__main__":
    main()
