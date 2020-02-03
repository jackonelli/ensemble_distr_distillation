"""Data loader for CIFAR data"""
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src import utils


class Cifar10Data:
    """CIFAR data wrapper
    Create instance like this:
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)
    """

    def __init__(self, ind=None, train=True, augmentation=False, transpose=False, root="./data"):
        self._log = logging.getLogger(self.__class__.__name__)

        self.transpose = transpose
        if augmentation:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

        else:
            self.transform = transforms.ToTensor()

        self.set = torchvision.datasets.CIFAR10(root=root,
                                                train=train,
                                                download=True)

        if ind is not None:
            self.set.data = np.array(self.set.data)[ind, :, :]
            self.set.targets = np.array(self.set.targets)[ind]

        self.input_size = self.set.data.shape[0]
        self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                        "horse", "ship", "truck")
        self.num_classes = len(self.classes)

    def __len__(self):
        return self.input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, ensemble_preds, ensemble_logits, target) where target is index of the target class.
        """
        img, target = self.set.data[index], self.set.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transpose:
            img = np.transpose(img, (1, 2, 0))
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(target)

        return img, target


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = Cifar10Data()

    bs = 4
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=bs,
                                         shuffle=True,
                                         num_workers=0)
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    plt.imshow(np.transpose(torchvision.utils.make_grid(images).numpy(), (1, 2, 0)))
    plt.show()

    # print labels
    print(" ".join("%5s" % data.classes[labels[j]] for j in range(bs)))


if __name__ == "__main__":
    main()
