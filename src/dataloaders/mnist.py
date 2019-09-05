import logging
import torchvision
import src.utils as utils
import torch
import matplotlib.pyplot as plt
import numpy as np


class MnistData:
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

    def __init__(self, root="./data", train=True):
        self._log = logging.getLogger(self.__class__.__name__)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(
            ),  # torchvision.transforms.Normalize((0.1307,), (0.3081,)
            utils.ReshapeTransform((-1,))
        ])

        self.set = torchvision.datasets.MNIST('./data_2',
                                              train=train,
                                              download=True,
                                              transform=transform)




def main():
    # get some random training images
    data = MnistData()
    loader = torch.utils.data.DataLoader(data.set,
                                         batch_size=4,
                                         shuffle=True,
                                         num_workers=1)
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join("%5s" % data.classes[labels[j]] for j in range(4)))


def imshow(img):
    img = img * 255
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()

