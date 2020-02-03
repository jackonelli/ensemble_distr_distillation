"""Data loader for CIFAR data with ensemble predictions"""
import logging
import torch
import h5py
import numpy as np


class Cifar10DataPredictions:
    """Saved predictions for (corrupted) CIFAR10 data, wrapper
    """

    def __init__(self, model, corruption, intensity, rep=None, data_dir="data/"):

        self._log = logging.getLogger(self.__class__.__name__)

        model_list = ["dropout", "dropout_nofirst", "ensemble", "ll_dropout", "ll_svi", "svi", "temp_scaling",
                      "vanilla"]
        corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                           "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "pixelate",
                           "saturate", "shot_noise", "spatter", "speckle_noise", "zoom_blur"]
        intensity_list = [0, 1, 2, 3, 4, 5]

        if (model not in model_list) or (intensity not in intensity_list) or (corruption not in corruption_list):
            print("Data not found: model, corruption or intensity does not exist")

        if rep is not None and rep > 4:
            print("Variable rep has to be between 0 and 4")

        elif model == "ensemble":

            filepath = "ensemble_predictions/ensemble_predictions.h5"

            with h5py.File(filepath, 'r') as f:

                if intensity == 0:
                    sub_grp = f["test"]
                else:
                    grp = f[corruption]
                    sub_grp = grp["intensity_" + str(intensity)]

                self.predictions = sub_grp["predictions"][()]
                self.targets = sub_grp["targets"][()]

                if rep is not None:
                    ensemble_size = 10
                    self.predictions = self.predictions[:, rep*ensemble_size:(rep+1)*ensemble_size, :]
                    self.targets = self.targets[:, rep*ensemble_size:(rep+1)*ensemble_size, :]

        else:

            filepath = data_dir + "cifar_model_predictions.hdf5"

            with h5py.File(filepath, 'r') as f:
                grp = f[model]

                if intensity == 0:
                    sub_grp = grp["test"]
                else:
                    sub_grp = grp["corrupt-static-" + corruption + "-" + str(intensity)]

                #for key in data_set_item.keys()
                predictions = sub_grp["probs"][()]
                targets = sub_grp["labels"][()]

                if rep is None:
                    self.predictions = predictions.reshape(predictions.shape[0]*predictions.shape[1], predictions.shape[2])
                    self.targets = targets.reshape(targets.shape[0]*targets.shape[1])
                else:
                    self.predictions = predictions[rep, :, :]
                    self.targets = targets[rep, :]

            self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                            "horse", "ship", "truck")
            self.num_classes = len(self.classes)

            self.length = self.predictions.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (prediction, label) where label is index of the target class.
        """
        preds = torch.tensor(self.predictions[index, :])
        target = torch.tensor(self.targets[index], dtype=torch.int64)

        return preds, target


def main():
    """Entry point for debug visualisation"""
    # get some random training images
    data = Cifar10DataPredictions("dropout", "brightness", 1)
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=4,
                                         shuffle=False,
                                         num_workers=0)
    dataiter = iter(loader)
    predictions, targets = dataiter.next()
    acc = np.mean(np.argmax(predictions.data.numpy(), axis=-1) == targets.data.numpy())
    print("Accuracy is: {}".format(acc))


if __name__ == "__main__":
    main()
