import logging
import torchvision
import src.utils as utils
import numpy as np


class MnistData():
    """MNIST data wrapper
    Create instance like this:
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=1)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=1)
    """

    def __init__(self, train=True, data_set="train", root="./data"):
        self._log = logging.getLogger(self.__class__.__name__)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        self.set = torchvision.datasets.MNIST(root=root,
                                              train=train,
                                              download=True,
                                              transform=self.transform)

        if train and data_set == 'train':
            self.data = self.set.data[:50000, :, :]
            self.targets = self.set.targets[:50000]

        elif train and data_set == 'validation':
            self.data = self.set.data[50000:, :, :]
            self.targets = self.set.targets[50000:]
        else:
            self.data = self.set.data
            self.targets = self.set.targets

        self.n_samples = self.data.size(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = self.transform(np.array(self.data[index], dtype=np.float32))
        x = x.reshape(-1) / 255
        y = np.array(self.targets[index], dtype=np.long)

        return x, y

    def get_sample(self, index):
        return self.__getitem__(index)

