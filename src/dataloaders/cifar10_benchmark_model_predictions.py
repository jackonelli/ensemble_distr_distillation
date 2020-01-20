"""Data loader for CIFAR data with ensemble predictions"""
import logging
import torch
import numpy as np
import h5py


class Cifar10Data:
    """CIFAR data with corruptions, wrapper
    """

    def __init__(self, model, corruption, intensity, rep=None, filedir=""):

        self._log = logging.getLogger(self.__class__.__name__)

        model_list = ["dropout", "dropout_nofirst", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling",
                      "vanilla"]
        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                           "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "pixelate",
                           "saturate", "shot_noise", "spatter", "speckle_noise", "zoom_blur"]
        intensity_list = [0, 1, 2, 3, 4, 5]

        filepath = filedir + "cifar_model_predictions.hdf5"

        if (model not in model_list) or (intensity not in intensity_list) or (corruption not in corruption_list):
            self._log.info("Data not found: model, corruption or intensity does not exist")

        else:

            with h5py.File(filepath, 'r') as f:
                model_item = f[model]

                if intensity == 0:
                    data_set_item = model_item["test"]
                else:
                    data_set_item = model_item["corrupt-static-" + corruption + "-" + str(intensity)]

                #for key in data_set_item.keys()
                predictions = data_set_item["probs"][()]
                labels = data_set_item["labels"][()]

            if rep is None:
                predictions = predictions.reshape(predictions.shape[0]*predictions.shape[1], predictions.shape[2])
                labels = labels.reshape(labels.shape[0]*labels.shape[1])
            else:
                predictions = predictions[rep, :, :]
                labels = labels[rep, :]

            self.set = CustomSet(predictions, labels)

            self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                            "horse", "ship", "truck")
            self.num_classes = len(self.classes)


class CustomSet:

    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels
        self.input_size = self.predictions.shape[0]

    def __len__(self):
        return self.input_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (prediction, label) where label is index of the target class.
        """
        preds = torch.tensor(self.predictions[index, :])
        target = torch.tensor(self.labels[index], dtype=torch.int64)

        return preds, target


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = Cifar10Data("dropout", "brightness", 1)
    loader = torch.utils.data.DataLoader(data.set,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)
    dataiter = iter(loader)
    labels, predictions = dataiter.next()


if __name__ == "__main__":
    main()
