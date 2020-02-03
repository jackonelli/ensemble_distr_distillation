import numpy as np
import logging
import src.ensemble.tensorflow_ensemble as tensorflow_ensemble
import tensorflow as tf
import torch
import src.dataloaders.cifar10_corrupted as cifar10_corrupted
import src.dataloaders.cifar10 as cifar10
import h5py
from pathlib import Path
from datetime import datetime
from src import utils


LOGGER = logging.getLogger(__name__)


def ensemble_predictions():
    # Will load ensemble, make and save predictions here
    train_set = cifar10.Cifar10Data(transpose=True)
    test_set = cifar10.Cifar10Data(train=False, transpose=True)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    models_dir = "../models/cifar-0508-ensemble_50/r"
    ensemble.load_ensemble(models_dir, num_members=50)

    data_list = [train_set, test_set]
    labels = ["train",  "test"]

    data_dir = "../../dataloaders/data/ensemble_predictions/"
    hf = h5py.File(data_dir + 'ensemble_predictions.h5', 'w')

    for data_set, label in zip(data_list, labels):
        data, logits, predictions, targets = [], [], [], []

        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=0)

        for batch in data_loader:
            inputs, labels = batch
            data.append(inputs)
            targets.append(labels)

            logs, preds = ensemble.predict(tf.convert_to_tensor(inputs.data.numpy()))
            logits.append(logs)
            predictions.append(preds)

        data = tf.concat(data, axis=0)
        logits = tf.concat(logits, axis=0)
        predictions = tf.concat(predictions, axis=0)
        targets = tf.concat(targets, axis=0)

        # Some kind of sanity check
        preds = np.argmax(np.mean(predictions.numpy(), axis=1), axis=-1)
        acc = np.mean(preds == np.array(data_set.set.targets))
        LOGGER.info("Accuracy on {} data set is: {}".format(label, acc))

        grp = hf.create_group(label)
        grp.create_dataset("data", data.numpy())
        grp.create_dataset("logits", logits.numpy())
        grp.create_dataset("predictions", predictions.numpy())
        grp.create_dataset("targets", targets.numpy())

    return logits, predictions


def ensemble_predictions_corrupted_data():

    # Load model
    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    models_dir = "../models/cifar-0508-ensemble_50/r"
    ensemble.load_ensemble(models_dir, num_members=50)

    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "motion_blur", "pixelate", "saturate",
                       "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
    data_set_size = 10000

    data_dir = "../../dataloaders/data/ensemble_predictions/"
    hf = h5py.File(data_dir + 'ensemble_predictions.h5', 'w')

    for i, corruption in enumerate(corruption_list):
        corr_grp = hf.create_group(corruption)

        # Load the data
        data = cifar10_corrupted.Cifar10DataCorrupted(corruption=corruption, data_dir="../dataloaders/data/CIFAR-10-C/",
                                                      torch=False)
        dataloader = torch.utils.data.DataLoader(data.set,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)

        data = []
        predictions = []
        targets = []
        intensity = 1
        for j, batch in enumerate(dataloader):
            inputs, labels = batch

            targets.append(tf.convert_to_tensor(labels.data.numpy()))

            inputs = tf.convert_to_tensor(inputs.data.numpy())
            data.append(inputs)
            logits, preds = ensemble.predict(inputs)
            predictions.append(preds)

            if (j * dataloader.batch_size) == data_set_size:
                sub_grp = corr_grp.create_group("intensity_" + str(intensity))

                data = tf.concat(data, axis=0).numpy()
                sub_grp.create_dataset("data", data)

                predictions = tf.concat(predictions, axis=0).numpy()
                sub_grp.create_dataset("predictions", predictions)

                targets = tf.concat(targets, axis=0).numpy()
                sub_grp.create_dataset("targets", targets)

                preds = np.argmax(np.mean(predictions, axis=1), axis=-1)
                acc = np.mean(preds == targets)
                LOGGER.info("Accuracy on {} data set with intensity {} is {}".format(corruption, intensity, acc))

                data = []
                predictions = []
                targets = []
                intensity += 1

    hf.close()


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))
    ensemble_predictions()


if __name__ == "__main__":
    main()

