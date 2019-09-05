import logging
import torchvision
import utils


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

    def __init__(self, root="./data", train=True):
        self._log = logging.getLogger(self.__class__.__name__)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(
            ),  # torchvision.transforms.Normalize((0.1307,), (0.3081,)
            utils.ReshapeTransform((-1,))
        ])

        self.set = torchvision.datasets.MNIST('./data',
                                              train=train,
                                              download=True,
                                              transform=transform)
