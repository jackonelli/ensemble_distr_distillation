import numpy as np
import logging
import torch
from src import utils
from src import metrics
from pathlib import Path
from datetime import datetime
from src.dataloaders import cifar10_corrupted
from src.dataloaders import cifar10_ensemble_pred
from src.ensemble import tensorflow_ensemble
from src.distilled import cifar_resnet_logits
from src import resnet_utils
import h5py

LOGGER = logging.getLogger(__name__)


def train_distilled_network():
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    N = 50000
    train_ind = np.random.choice(np.arange(0, N), size=40000, replace=False)
    valid_ind = np.stack([i for i in np.arange(N) if i not in train_ind])

    train_set = cifar10_ensemble_pred.Cifar10Data(ind=train_ind, augmentation=True)
    valid_set = cifar10_ensemble_pred.Cifar10Data(ind=valid_ind)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=2)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=2)

    test_set = cifar10_ensemble_pred.Cifar10Data(train=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=2)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(
        output_size=10)  # Borde kanske köra med bara 10 medlemmar??? Kan jag ha det som en inparameter kanske,
    # Typ ensemble indices
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    distilled_model = cifar_resnet_logits.ResnetLogitsTeacherFromFile(ensemble,
                                                                      resnet_utils.Bottleneck,
                                                                      [2, 2, 2, 2],
                                                                      learning_rate=args.lr * 0.1,
                                                                      scale_teacher_logits=True)

    loss_metric = metrics.Metric(name="Mean loss", function=distilled_model.calculate_loss)
    acc_metric = metrics.Metric(name="Mean rel acc", function=metrics.accuracy_logits)
    distilled_model.add_metric(loss_metric)
    distilled_model.add_metric(acc_metric)

    distilled_model.train(train_loader, num_epochs=200, validation_loader=valid_loader)

    distilled_model.eval_mode()
    counter = 0
    model_acc = 0

    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(distilled_model.device)

        predicted_distribution = distilled_model.predict(inputs)
        predicted_distribution = torch.cat(predicted_distribution, 1 - torch.sum(predicted_distribution, dim=-1),
                                           dim=-1)
        model_acc += metrics.accuracy(predicted_distribution.to(distilled_model.device), labels.int())
        counter += 1

    torch.save(distilled_model.state_dict(), "../models/distilled_model_cifar10")


def check_trained_model(file_dir="../models/distilled_model_cifar10"):
    args = utils.parse_args()

    test_set = cifar10_ensemble_pred.Cifar10Data(train=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=0)

#    distilled_model = torch.load("models/distilled_model_cifar10")

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    distilled_model = cifar_resnet_logits.ResnetLogitsTeacherFromFile(ensemble,
                                                                      resnet_utils.Bottleneck,
                                                                      [2, 2, 2, 2],
                                                                      learning_rate=args.lr * 0.1,
                                                                      scale_teacher_logits=True)

    distilled_model.load_state_dict(torch.load(file_dir))
    distilled_model.eval_mode()

    counter = 0
    distilled_model_acc = 0
    for batch in test_loader:
        inputs, labels = batch

        distilled_distribution = distilled_model.predict(inputs)
        distilled_distribution = torch.mean(torch.cat((distilled_distribution,
                                                       1 - torch.sum(distilled_distribution, dim=-1, keepdim=True)),
                                                      dim=-1), dim=1)
        distilled_model_acc += metrics.accuracy(distilled_distribution, labels.int())
        counter += 1

    distilled_model_acc = distilled_model_acc / counter
    LOGGER.info("Distilled model accuracy on {} data {}".format("test", distilled_model_acc))


def predictions(file_dir="../models/distilled_model_cifar10"):
    args = utils.parse_args()

    train_set = cifar10_ensemble_pred.Cifar10Data()
    test_set = cifar10_ensemble_pred.Cifar10Data(train=False)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    distilled_model = cifar_resnet_logits.ResnetLogitsTeacherFromFile(ensemble,
                                                                      resnet_utils.Bottleneck,
                                                                      [2, 2, 2, 2],
                                                                      learning_rate=args.lr * 0.1,
                                                                      scale_teacher_logits=True)

    distilled_model.load_state_dict(torch.load(file_dir=file_dir))

    data_list = [train_set, test_set]
    labels = ["train",  "test"]

    data_dir = "../../dataloaders/data/"
    hf = h5py.File(data_dir + 'distilled_model_predictions.h5', 'w')

    for data_set, label in zip(data_list, labels):
        data, logits, predictions, targets = [], [], [], []

        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=0)

        for batch in data_loader:
            inputs, labels = batch
            img = inputs[0].to(distilled_model.device)
            data.append(img.data.numpy())
            targets.append(labels.data.numpy())

            logs, probs = distilled_model.predict(img, return_logits=True)
            logits.append(logs.data.numpy())
            predictions.append(probs.data.numpy())

        data = np.concat(data, axis=0)
        logits = np.concat(logits, axis=0)
        predictions = np.concat(predictions, axis=0)
        targets = np.concat(targets, axis=0)

        # Some kind of sanity check
        preds = np.argmax(np.mean(predictions.numpy(), axis=1), axis=-1)
        acc = np.mean(preds == targets)
        LOGGER.info("Accuracy on {} data set is: {}".format(label, acc))

        grp = hf.create_group(label)
        grp.create_dataset("data", data)
        grp.create_dataset("logits", logits)
        grp.create_dataset("predictions", predictions)
        grp.create_dataset("targets", targets)

    return logits, predictions


def predictions_corrupted_data(file_dir="../models/distilled_model_cifar10"):
    args = utils.parse_args()

    # Load model
    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    distilled_model = cifar_resnet_logits.ResnetLogitsTeacherFromFile(ensemble,
                                                                      resnet_utils.Bottleneck,
                                                                      [2, 2, 2, 2],
                                                                      learning_rate=args.lr * 0.1,
                                                                      scale_teacher_logits=True)

    distilled_model.load_state_dict(torch.load(file_dir = "../models/distilled_model_cifar10"))

    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "motion_blur", "pixelate", "saturate",
                       "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
    data_set_size = 10000

    data_dir = "../../dataloaders/data/"
    hf = h5py.File(data_dir + 'distilled_model_predictions.h5', 'w')

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

            data.append(inputs)
            preds = distilled_model.predict(inputs)
            predictions.append(preds)

            if (j * dataloader.batch_size) == data_set_size:
                sub_grp = corr_grp.create_group("intensity_" + str(intensity))

                data = np.concat(data, axis=0)
                sub_grp.create_dataset("data" + str(intensity), data)

                predictions = np.concat(predictions, axis=0)
                sub_grp.create_dataset("predictions", predictions)

                targets = np.concat(targets, axis=0)
                sub_grp.create_dataset("targets", targets)

                preds = np.argmax(np.mean(predictions, axis=1), axis=-1)
                acc = np.mean(preds == targets)
                print("Accuracy on {} data set with intensity {} is {}".format(corruption, intensity, acc))

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
    train_distilled_network()


if __name__ == "__main__":
    main()
