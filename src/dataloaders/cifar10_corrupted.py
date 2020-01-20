"""Data loader for CIFAR data with ensemble predictions"""
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Cifar10Data:
    """CIFAR data with corruptions, wrapper
    """

    def __init__(self, corruption, data_dir="data/CIFAR-10-C/", torch=True):
        self._log = logging.getLogger(self.__class__.__name__)

        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                           "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
                           "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
                           "zoom_blur"]

        if corruption not in corruption_list:
            self._log.info("Data not found: corruption does not exist")

        else:

            data = np.load(data_dir + corruption + ".npy")
            labels = np.load(data_dir + "labels.npy")

            self.set = CustomSet(data, labels, torch=torch)

            self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                            "horse", "ship", "truck")
            self.num_classes = len(self.classes)


class CustomSet:

    def __init__(self, data, labels, torch=True):
        self.data = data
        self.labels = labels
        self.input_size = self.data.shape[0]
        self.torch = torch

    def __len__(self):
        return self.input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.torch:
            img = transforms.ToTensor()(Image.fromarray(img))
        else:
            img = img / 255

        target = self.labels[index]

        return img, target


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = Cifar10Data(corruption="brightness")
    loader = torch.utils.data.DataLoader(data.set,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)
    dataiter = iter(loader)
    inputs, labels = dataiter.next()

    img = inputs[0]

    # show images
    imshow(torchvision.utils.make_grid(img))
    # print labels
    print(" ".join("%5s" % data.classes[labels[j]] for j in range(4)))


def imshow(img):
    """Imshow helper
    TODO: Move to utils
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    main()
