import logging
from pathlib import Path
from datetime import datetime
from src import utils
import numpy as np
import tensorflow as tf
import torch
import h5py

from src.dataloaders import cifar10, cifar10_corrupted
from src.experiments.cifar10 import resnet_utils
from src.ensemble import ensemble
from src.ensemble import cifar_resnet
from src import metrics

LOGGER = logging.getLogger(__name__)


def ensemble_predictions(ensemble, torch_data=True):  # If torch_data is not true, ensemble is assumed to be Tensorflow
    """ Make and save predictions from ensemble """
    train_set = cifar10.Cifar10Data(torch_data=torch_data)
    test_set = cifar10.Cifar10Data(train=False, torch_data=torch_data)

    data_list = [test_set, train_set]
    labels = ["test", "train"]

    data_dir = "../../dataloaders/data/ensemble_predictions/"
    hf = h5py.File(data_dir + 'ensemble_predictions.h5', 'w')

    for data_set, label in zip(data_list, labels):
        data, logits, predictions, targets = [], [], [], []

        data_loader = torch_data.utils.data.DataLoader(data_set,
                                                       batch_size=100,
                                                       shuffle=False,
                                                       num_workers=0)

        for batch in data_loader:
            inputs, labels = batch

            if not torch_data:
                inputs = tf.convert_to_tensor(inputs.data.numpy())
                targets.append(tf.convert_to_tensor(labels.data.numpy()))

            data.append(inputs)
            logs, preds = ensemble.predict(inputs)
            logits.append(logs)
            predictions.append(preds)

        if torch_data:
            data = np.concatenate(data.data.numpy(), axis=0)
            logits = np.concatenate(logits.data.numpy(), axis=0)
            predictions = np.concatenate(predictions.data.numpy(), axis=0)
            targets = np.concatenate(targets.data.numpy(), axis=0)

        else:
            data = tf.concat(data, axis=0).numpy()
            logits = tf.concat(logits, axis=0).numpy()
            predictions = tf.concat(predictions, axis=0).numpy()
            targets = tf.concat(targets, axis=0).numpy()

        # Check accuracy
        preds = np.argmax(np.mean(predictions, axis=1), axis=-1)
        acc = np.mean(preds == np.array(data_set.set.targets))
        LOGGER.info("Accuracy on {} data set is: {}".format(label, acc))

        grp = hf.create_group(label)
        grp.create_dataset("data", data=data)
        grp.create_dataset("logits", data=logits)
        grp.create_dataset("predictions", data=predictions)
        grp.create_dataset("targets", data=targets)


def ensemble_predictions_corrupted_data(ensemble, torch_data=True):
    """ Make and save predictions from ensemble on corrupted data sets"""

    # Load model
    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "motion_blur", "pixelate", "saturate",
                       "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
    intensity_list = [1, 2, 3, 4, 5]

    data_dir = "../../dataloaders/data/ensemble_predictions/"
    hf = h5py.File(data_dir + 'ensemble_predictions_corrupted_data.h5', 'w')

    for i, corruption in enumerate(corruption_list):
        corr_grp = hf.create_group(corruption)

        data = []
        predictions = []
        logits = []
        targets = []

        for intensity in intensity_list:
            # Load the data
            data_set = cifar10_corrupted.Cifar10DataCorrupted(corruption=corruption, intensity=intensity, torch_data=False)
            dataloader = torch.utils.data.DataLoader(data_set.set,
                                                     batch_size=100,
                                                     shuffle=False,
                                                     num_workers=2)
            for batch in enumerate(dataloader):
                inputs, labels = batch

                if not torch_data:
                    targets.append(tf.convert_to_tensor(labels.data.numpy()))
                    inputs = tf.convert_to_tensor(inputs.data.numpy())

                data.append(inputs)
                logs, preds = ensemble.predict(inputs)
                predictions.append(preds)
                logits.append(logs)

                sub_grp = corr_grp.create_group("intensity_" + str(intensity))

                if torch_data:
                    data = np.concatenate(data.data.numpy(), axis=0)
                    predictions = np.concatenate(predictions.data.numpy(), axis=0)
                    logits = np.concatenate(logits.data.numpy(), axis=0)
                    targets = np.concatenate(targets.data.numpy(), axis=0)
                else:
                    data = tf.concat(data, axis=0).numpy()
                    predictions = tf.concat(predictions, axis=0).numpy()
                    logits = tf.concat(logits, axis=0).numpy()
                    targets = tf.concat(targets, axis=0).numpy()

                sub_grp.create_dataset("data", data=data)
                sub_grp.create_dataset("predictions", data=predictions)
                sub_grp.create_dataset("logits", data=logits)
                sub_grp.create_dataset("targets", data=targets)

                preds = np.argmax(np.mean(predictions, axis=1), axis=-1)
                acc = np.mean(preds == targets)
                LOGGER.info("Accuracy on {} data set with intensity {} is {}".format(corruption, intensity, acc))

                data = []
                predictions = []
                logits = []
                targets = []

    hf.close()


def train_ensemble(args, ensemble_filepath="models/resnet_ensemble"):

    output_size = 10
    prob_ensemble = ensemble.Ensemble(output_size=output_size)

    data_ind = np.load("data/training_data_indices.npy")
    num_train_points = 40000
    train_ind = data_ind[:num_train_points]
    valid_ind = data_ind[num_train_points:]

    train_set = cifar10.Cifar10Data(ind=train_ind, augmentation=True)
    valid_set = cifar10.Cifar10Data(ind=valid_ind)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    # test_set = cifar10.Cifar10Data(train=False)
    #
    # test_loader = torch.utils.data.DataLoader(test_set,
    #                                           batch_size=64,
    #                                           shuffle=True,
    #                                           num_workers=0)

    device = utils.torch_settings(args.seed, args.gpu)
    for _ in range(1): # args.num_ensemble_members
        resnet_model = cifar_resnet.ResNet(resnet_utils.Bottleneck, [2, 2, 2, 2], device=device, learning_rate=args.lr)

        resnet_model.train(train_loader, validation_loader=valid_loader, num_epochs=1, # args.num_epochs
                           reshape_targets=False)

        prob_ensemble.add_member(resnet_model)

    loss_metric = metrics.Metric(name="Loss", function=torch.nn.CrossEntropyLoss)
    prob_ensemble.add_metrics([loss_metric])
    prob_ensemble.train(train_loader, args.num_epochs, validation_loader=valid_loader)

    prob_ensemble.save_ensemble(ensemble_filepath)

    return ensemble


def load_ensemble():
    """Placeholder for loading ensemble from elsewhere"""
    pass


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))

    ensemble = train_ensemble(args)
    #ensemble_predictions_corrupted_data(ensemble)
    #ensemble_predictions(ensemble)


if __name__ == "__main__":
    main()

