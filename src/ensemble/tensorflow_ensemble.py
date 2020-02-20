import logging
import tensorflow as tf
from scipy.special import softmax as scipy_softmax
from src.experiments.cifar10.uq_benchmark_2019 import models_lib
from src.experiments.cifar10.uq_benchmark_2019 import experiment_utils

from src.dataloaders import cifar10
import h5py
import numpy as np


"""This is a wrapper class that extracts saved data from a specific type of dataset"""
class TensorflowEnsemble:
    def __init__(self, output_size, indices=None):
        """The ensemble member needs to track the size
        of the output of the ensemble
        This can be automatically inferred but it would look ugly
        and this now works as a sanity check as well
        """
        self.members = list()
        self._log = logging.getLogger(self.__class__.__name__)
        self.output_size = output_size
        self.size = 0
        self.indices = indices

    def add_member(self, new_member):
            self._log.info("Adding {} to ensemble".format(type(new_member)))
            self.members.append(new_member)
            self.size += 1

    def add_multiple(self, number_of, constructor):
        for _ in range(number_of):
            self.add_member(constructor())

    def train(self, train_loader, num_epochs, validation_loader=None):
        # We will not have this option for now
        pass

    def get_predictions(self, inputs):
        # Inputs should be a tuple (x, ensemble predictions, ensemble logits)

        predictions = inputs[1]

        if self.indices is not None:
            predictions = predictions[:, self.indices, :]

        return predictions

    def get_logits(self, inputs):
        # Inputs should be a tuple (x, ensemble predictions, ensemble logits)

        logits = inputs[2]

        if self.indices is not None:
            logits = logits[:, self.indices, :]

        return logits

    def predict(self, inputs):
        """Ensemble prediction
        Returns the predictions of all individual ensemble members.
        B = batch size, K = num output params, N = ensemble size


        Args:
            inputs (torch.tensor((B, data_dim))): data batch

        Returns:
            predictions (torch.tensor((B, N, K)))
        """
        logits = []
        predictions = []
        for member_ind, member in enumerate(self.members):
            #logits.append(member.predict(inputs))
            stats = experiment_utils.make_predictions(member, inputs)
            logits.append(np.squeeze(stats['logits_samples']))
            predictions.append(stats['probs'])

        #predictions = tf.stack([scipy_softmax(x, axis=-1) for x in logits], axis=1)
        logits = tf.stack(logits, axis=1)
        predictions = tf.stack(predictions, axis=1)

        return logits, predictions

    def eval_mode(self):

        for member in self.members:
            for layer in member.layers:
                layer.trainable = False  # TODO: check for sublayers

    def load_ensemble(self, models_dir, num_members=None):

        for i in range(num_members):
            member = models_lib.load_model(models_dir + str(i))
            self.add_member(member)


def test_predict():
    ensemble = TensorflowEnsemble(output_size=10)

    models_dir = "../experiments/models/cifar-0508-ensemble_50/r"
    ensemble.load_ensemble(models_dir, num_members=1)
    ensemble.eval_mode()

    train_set = cifar10.Cifar10Data(train=True, torch=False)
    train_data = train_set.set.data / 255
    mean = (np.mean(train_data[:, :, :, 0]), np.mean(train_data[:, :, :, 1]), np.mean(train_data[:, :, :, 2]))
    std = (np.std(train_data[:, :, :, 0]), np.std(train_data[:, :, :, 1]), np.std(train_data[:, :, :, 2]))
    print(mean)
    print(std)

    test_set = cifar10.Cifar10Data(train=False, torch=False)

    num_samples = 3
    data_points = test_set.set.data[:num_samples, :, :, :] / 255
    labels = test_set.set.targets[:num_samples]

    intensity = 5
    base = (intensity - 1) * 10000 + 100
    corruption = "brightness"
    data = np.load("../dataloaders/data/CIFAR-10-C/" + corruption + ".npy")
    data_points = data[base:(base + num_samples), :, :, :].astype("float64") / 255

    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #for i in np.arange(3):
    #    data_points[:, :, :, i] = (data_points[:, :, :, i] - mean[i]) / mean[i]

    ens_targets = np.load("../dataloaders/data/CIFAR-10-C/labels.npy")
    labels = ens_targets[:num_samples]
    print(labels)

    dataset = tf.data.Dataset.from_tensor_slices((data_points, labels)).batch(num_samples)
    logits, predictions = ensemble.predict(dataset)

    new_preds = np.mean(predictions, axis=1)
    print("New {}".format(new_preds))
    print(np.argmax(new_preds, axis=-1))

    # Compare with
    filepath = "../dataloaders/data/cifar_model_predictions.hdf5"

    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                       "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "pixelate",
                       "saturate", "shot_noise", "spatter", "speckle_noise", "zoom_blur"]

    for corruption in corruption_list:
        print(corruption)

        for intensity in np.arange(1, 6):
            print(intensity)
            with h5py.File(filepath, 'r') as f:
                grp = f["ensemble"]
                #sub_grp = grp["test"]
                sub_grp = grp["corrupt-static-" + corruption + "-" + str(intensity)]

                predictions = sub_grp["probs"][()]
                targets = sub_grp["labels"][()]

            ens_pred = np.load("../dataloaders/data/ensemble_predictions/" + corruption + ".npy")
            ens_distr = np.mean(ens_pred[((intensity - 1) * 10000):(intensity * 10000), :, :], axis=1)
            print(np.mean(np.argmax(ens_distr, axis=-1) == labels))
            distr = np.mean(predictions, axis=0)
            print(np.mean(np.argmax(distr, axis=-1) == targets))
            t = np.sum(np.abs(ens_distr - distr))
            print(t)

    print(targets[0, 100:(100+num_samples)])
    old_preds = np.mean(predictions[:, 100:(100+num_samples), :], axis=0)
    print("Old {}".format(old_preds))
    print(np.argmax(old_preds, axis=-1))


if __name__ == "__main__":
   test_predict()


