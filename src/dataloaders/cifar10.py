"""Data loader for CIFAR data"""
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, root="./data", train=True):
        self._log = logging.getLogger(self.__class__.__name__)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.set = torchvision.datasets.CIFAR10(root=root,
                                                train=train,
                                                download=True,
                                                transform=transform)

        self.input_size = None
        self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                        "horse", "ship", "truck")
        self.num_classes = len(self.classes)


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = Cifar10Data()
    loader = torch.utils.data.DataLoader(data.set,
                                         batch_size=4,
                                         shuffle=True,
                                         num_workers=2)
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join("%5s" % data.classes[labels[j]] for j in range(4)))


def imshow(img):
    """Imshow helper
    TODO: Move to utils
    """

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
