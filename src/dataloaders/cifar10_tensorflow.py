from tensorflow.keras.utils import Sequence
import tensorflow.keras.datasets.cifar10 as cifar10
import torchvision
import numpy as np
from abc import ABC, abstractmethod
import torch


class CustomDataGenerator(Sequence, ABC):  # The inheritance is not really necessary anymore
    """Skeleton for data loaders, the subclasses should themselves initialize the data"""

    def __init__(self, batch_size, train=True):
        self.train = train
        self.batch_size = batch_size
        self.next_batch = False
        self.idx = 0

    def __len__(self):
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        self.batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[self.batch_indices]
        batch_y = self.y[self.batch_indices]

        return batch_x, batch_y

    def get_next_batch(self):
        batch_x, batch_y = self.__getitem__(self.idx)

        if (self.idx + 1) * self.batch_size == self.N:
            self.next_batch = False

        self.idx += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def reset_batch(self):
        #self.on_epoch_end(), will skip this
        self.next_batch = True
        self.idx = 0

    def save_data(self, x_filepath, y_filepath):
        np.save(x_filepath, self.x)
        np.save(y_filepath, self.y)

    @abstractmethod
    def _initialize_data(self):
        pass


class Cifar10(CustomDataGenerator):
    def __init__(self, batch_size=100, train=True):  # train is used to load either train or test data

        super().__init__(batch_size=batch_size, train=train)
        self._initialize_data()

    def _initialize_data(self):
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
        dataset = torchvision.datasets.CIFAR10(root="./data", train=self.train, download=True)

        self.x = dataset.data / 255  # TODO: CHECK FOR TRANSFORMATIONS
        self.y = np.asarray(dataset.targets)

        self.N = self.x.shape[0]
        self.indices = np.arange(self.N)
        self.reset_batch()


