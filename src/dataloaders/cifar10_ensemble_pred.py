"""Data loader for CIFAR data with ensemble predictions"""
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Cifar10Data:
    """CIFAR data wrapper with ensemble predictions,
    data is organized as ((img, ensemble preds, ensemble logits), labels)

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

    def __init__(self, root="./data", data_dir="../dataloaders/data/ensemble_predictions/", train=True, validation=False):
        self._log = logging.getLogger(self.__class__.__name__)

        self.transform = None
        self.target_transform = None

        # Note that this is dependent upon that the ensemble predictions are loaded in the same order as the data.
        # TODO save ensemble predictions together with data
        self.set = torchvision.datasets.CIFAR10(root=root,
                                                train=train,
                                                download=True,
                                                transform=self.transform)

        if train:
            ensemble_preds = np.load(data_dir + "cifar10_ensemble_train_predictions.npy")
            ensemble_logits = np.load(data_dir + "cifar10_ensemble_train_logits.npy")

            split = 40000
            if validation:
                self.set.data = (self.set.data[split:, :, :, :], ensemble_preds[split:, :, :], ensemble_logits[split:, :, :])
                self.set.targets = self.set.targets[split:]
            else:
                self.set.data = (self.set.data[:1, :, :, :], ensemble_preds[:10, :, :], ensemble_logits[:1, :, :])
                self.set.targets = self.set.targets[:1]
        else:
            ensemble_preds = np.load(data_dir + "cifar10_ensemble_test_predictions.npy")
            ensemble_logits = np.load(data_dir + "cifar10_ensemble_test_logits.npy")
            assert ensemble_preds.shape[0] == self.set.data.shape[0] and \
                   ensemble_logits.shape[0] == self.set.data.shape[0]
            self.set.data = (self.set.data, ensemble_preds, ensemble_logits)

        ensemble_predictions = np.argmax(np.mean(self.set.data[1], axis=1), axis=-1)
        acc = np.mean(ensemble_predictions == np.squeeze(self.set.targets))
        print("Ensemble accuracy: {}".format(acc))

        self.input_size = self.set.data[0].shape[0]
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
        img, preds, logits = self.set.data[0], self.set.data[1], self.set.data[2]
        img, preds, logits, target = img[index], torch.Tensor(preds[index]), torch.Tensor(logits[index]), \
                                     self.set.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = transforms.ToTensor()(Image.fromarray(img))

        if self.transform is not None:
            img = self.transform(img)
            preds = self.transform(preds)
            logits = self.transform(logits)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, preds, logits), target


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = Cifar10Data()
    loader = torch.utils.data.DataLoader(data.set,
                                         batch_size=4,
                                         shuffle=True,
                                         num_workers=0)
    dataiter = iter(loader)
    inputs, labels = dataiter.next()

    img = inputs[0]
    probs = inputs[1].data.numpy()
    preds = np.argmax(probs, axis=-1)

    acc = np.mean(preds == labels)

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
