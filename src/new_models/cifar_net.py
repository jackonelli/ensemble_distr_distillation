import torch
import torch.nn as nn
import torch.optim as optim
import ensemble
import loss as custom_loss
import numpy as np


class EnsembleNet(ensemble.EnsembleMember):
    def __init__(self, device=torch.device("cpu"), learning_rate=0.001):
        super().__init__(loss_function=nn.CrossEntropyLoss(), device=device)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = optim.SGD(self.parameters(),
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

    def calculate_loss(self, inputs, labels):
        outputs = self.forward(inputs)

        return self.loss(outputs, labels)


class DistilledNet(ensemble.EnsembleMember):
    """Not necessarily an ensemble member but can be used as one"""

    def __init__(self,
                 teacher,
                 device=torch.device("cpu"),
                 learning_rate=0.001):
        super().__init__(loss_function=nn.CrossEntropyLoss(), device=device)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = optim.SGD(self.parameters(),
                                   lr=learning_rate,
                                   momentum=0.9)
        self.to(self.device)
        self.teacher = teacher

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def calculate_loss(self, inputs, labels, t):
        outputs = self.forward(inputs)
        soft_targets = self.teacher.predict(inputs, t)

        loss = custom_loss.scalar_loss(outputs, soft_targets)

        if labels is not None and self.use_hard_labels:
            loss += self.loss(outputs, labels)

        return loss

    def train_epoch(self, train_loader, t, hard_targets=False):
        """Train single epoch"""
        running_loss = 0
        for batch in train_loader:
            self.optimizer.zero_grad()
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

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
