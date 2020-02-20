import numpy as np
import logging
import torch
import h5py
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
from src import utils
from src import metrics
from pathlib import Path
from datetime import datetime
from src.dataloaders import cifar10_corrupted
from src.dataloaders import cifar10_ensemble_pred
from src.ensemble import tensorflow_ensemble
from src.distilled import cifar_resnet_logits
from src import resnet_utils

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
        output_size=10)
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    distilled_model = cifar_resnet_logits.CifarResnetLogits(ensemble,
                                                            resnet_utils.Bottleneck,
                                                            [2, 2, 2, 2],
                                                            learning_rate=args.lr,
                                                            scale_teacher_logits=True)

    loss_metric = metrics.Metric(name="Mean loss", function=distilled_model.calculate_loss)
    distilled_model.add_metric(loss_metric)
    #acc_metric = metrics.Metric(name="Mean rel acc", function=metrics.accuracy_logits)
    #distilled_model.add_metric(acc_metric)

    distilled_model.train(train_loader, num_epochs=args.num_epochs, validation_loader=valid_loader)

    distilled_model.eval_mode()
    counter = 0
    model_acc = 0

    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(distilled_model.device)

        predicted_distribution = distilled_model.predict(inputs)
        model_acc += metrics.accuracy(predicted_distribution.to(distilled_model.device), labels.int())
        counter += 1

    torch.save(distilled_model.state_dict(), "../models/distilled_model_cifar10")


def predictions(model_dir="../models/distilled_model_cifar10",
                file_dir="../../dataloaders/data/distilled_model_predictions.h5", vanilla=False):
    args = utils.parse_args()

    train_set = cifar10_ensemble_pred.Cifar10Data()
    test_set = cifar10_ensemble_pred.Cifar10Data(train=False)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    distilled_model = cifar_resnet_logits.CifarResnetLogits(ensemble,
                                                            resnet_utils.Bottleneck,
                                                            [2, 2, 2, 2],
                                                            learning_rate=args.lr,
                                                            scale_teacher_logits=True,
                                                            vanilla_distillation=vanilla)

    distilled_model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    distilled_model.eval_mode()

    data_list = [test_set, train_set]
    labels = ["test", "train"]

    hf = h5py.File(file_dir, 'w')

    for data_set, label in zip(data_list, labels):

        if vanilla:
            data, pred_samples,  teacher_logits, teacher_predictions, targets = [], [], [], [], []
        else:
            data, pred_samples, mean, var, logits, teacher_logits, teacher_predictions, targets = \
                [], [], [], [], [], [], [], []

        data_loader = torch.utils.data.DataLoader(data_set.set,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  num_workers=0)

        for batch in data_loader:
            inputs, labels = batch
            img = inputs[0].to(distilled_model.device)
            data.append(img.data.numpy())
            targets.append(labels.data.numpy())
            teacher_logits.append(inputs[2].data.numpy())
            teacher_predictions.append(inputs[1].data.numpy())

            if vanilla:
                probs = distilled_model.predict(img)
            else:
                m, v, logs, probs = distilled_model.predict(img, return_params=True, return_logits=True)
                mean.append(m.data.numpy())
                var.append(v.data.numpy())
                logits.append(logs.data.numpy())
            pred_samples.append(probs.data.numpy())

        data = np.concatenate(data, axis=0)
        pred_samples = np.concatenate(pred_samples, axis=0)
        teacher_logits = np.concatenate(teacher_logits, axis=0)
        teacher_predictions = np.concatenate(teacher_predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        if vanilla:
            preds = np.argmax(pred_samples, axis=-1)
        else:
            mean = np.concatenate(mean, axis=0)
            var = np.concatenate(var, axis=0)
            logits = np.concatenate(logits, axis=0)
            preds = np.argmax(np.mean(pred_samples, axis=1), axis=-1)


        # Check accuracy
        acc = np.mean(preds == targets)
        LOGGER.info("Accuracy on {} data set is: {}".format(label, acc))

        # Check accuracy relative teacher
        teacher_preds = np.argmax(np.mean(teacher_predictions, axis=1), axis=-1)
        rel_acc = np.mean(preds == teacher_preds)
        LOGGER.info("Accuracy on {} data set relative teacher is: {}".format(label, rel_acc))

        grp = hf.create_group(label)
        grp.create_dataset("data", data=data)
        grp.create_dataset("predictions", data=pred_samples)
        grp.create_dataset("teacher-logits", data=teacher_logits)
        grp.create_dataset("teacher-predictions", data=teacher_predictions)
        grp.create_dataset("targets", data=targets)

        if not vanilla:
            grp.create_dataset("mean", data=mean)
            grp.create_dataset("var", data=var)
            grp.create_dataset("logits", data=logits)

    return pred_samples


def predictions_corrupted_data(model_dir="../models/distilled_model_cifar10",
                               file_dir="../../dataloaders/data/distilled_model_predictions.h5", vanilla=False):
    args = utils.parse_args()

    # Load model
    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    distilled_model = cifar_resnet_logits.CifarResnetLogits(ensemble,
                                                            resnet_utils.Bottleneck,
                                                            [2, 2, 2, 2],
                                                            learning_rate=args.lr * 0.1,
                                                            scale_teacher_logits=True,
                                                            vanilla_distillation=vanilla)

    distilled_model.load_state_dict(torch.load(model_dir))

    distilled_model.eval_mode()

    corruption_list = ["test", "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                       "gaussian_blur",
                       "gaussian_noise", "glass_blur", "impulse_noise", "motion_blur", "pixelate", "saturate",
                       "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]


    hf = h5py.File(file_dir, 'w')

    for i, corruption in enumerate(corruption_list):
        corr_grp = hf.create_group(corruption)

        if corruption == "test":
            intensity_list = [0]
        else:
            intensity_list = [1, 2, 3, 4, 5]

        for intensity in intensity_list:
            # Load the data
            data_set = cifar10_corrupted.Cifar10DataCorrupted(corruption=corruption, intensity=intensity)
            dataloader = torch.utils.data.DataLoader(data_set.set,
                                                     batch_size=100,
                                                     shuffle=False,
                                                     num_workers=2)

            # data = []
            predictions = []
            targets = []

            if not vanilla:
                logits = []
                mean = []
                var = []
                raw_output = []

            for j, batch in enumerate(dataloader):
                inputs, labels = batch
                targets.append(labels.data.numpy())
                # data.append(inputs.data.numpy())

                inputs, labels = inputs.to(distilled_model.device), labels.to(distilled_model.device)

                if vanilla:
                    preds = distilled_model.predict(inputs)
                else:
                    m, v, raw, logs, preds = distilled_model.predict(inputs, return_raw_data=True, return_logits=True,
                                                                     comp_fix=True)
                    logits.append(logs.data.numpy())
                    mean.append(m.data.numpy())
                    var.append(v.data.numpy())
                    raw_output.append(raw.data.numpy())

                predictions.append(preds.to(torch.device("cpu")).data.numpy())

            sub_grp = corr_grp.create_group("intensity_" + str(intensity))

            # data = np.concatenate(data, axis=0)
            # sub_grp.create_dataset("data", data=data)

            predictions = np.concatenate(predictions, axis=0)
            sub_grp.create_dataset("predictions", data=predictions)

            targets = np.concatenate(targets, axis=0)
            sub_grp.create_dataset("targets", data=targets)

            if vanilla:
                preds = np.argmax(predictions, axis=-1)
            else:
                preds = np.argmax(np.mean(predictions, axis=1), axis=-1)

            acc = np.mean(preds == targets)
            print("Accuracy on {} data set with intensity {} is {}".format(corruption, intensity, acc))

            if not vanilla:
                logits = np.concatenate(logits, axis=0)
                sub_grp.create_dataset("logits", data=logits)

                mean = np.concatenate(mean, axis=0)
                sub_grp.create_dataset("mean", data=mean)

                var = np.concatenate(var, axis=0)
                sub_grp.create_dataset("var", data=var)

                raw_output = np.concatenate(raw_output, axis=0)
                sub_grp.create_dataset("raw_output", data=raw_output)

    hf.close()


def evaluate_parameters(data_set, data_label="", file_dir="../../dataloaders/data/distilled_model_predictions.h5",
                        teacher_ind=None):
    """Comparison of mean and var histograms of ensemble and distilled model"""

    # Loading predictions from file
    with h5py.File(file_dir, 'r') as f:
        data_item = f[data_set]
        distilled_model_mean = data_item["mean"][()]
        distilled_model_var = data_item["var"][()]
        ensemble_logits = data_item["teacher-logits"][()]

    if teacher_ind is not None:
        ensemble_logits = ensemble_logits[:, teacher_ind, :]

    max_value = 1000
    num_inf = np.sum(np.isinf(distilled_model_var))
    LOGGER.info("Setting {} infinite variance(s) to {}.".format(num_inf, max_value))
    distilled_model_var[np.isinf(distilled_model_var)] = max_value

    scaled_ensemble_logits = ensemble_logits - ensemble_logits[:, :, -1][:, :, np.newaxis]
    ensemble_mean = np.mean(scaled_ensemble_logits, axis=1)
    ensemble_var = np.var(scaled_ensemble_logits, axis=1)

    metric_list = [[ensemble_mean, distilled_model_mean], [ensemble_var, distilled_model_var]]
    metric_labels = ["mean", "variance"]

    num_bins = 100

    for metric, metric_label in zip(metric_list, metric_labels):
        ensemble_metric = metric[0]
        distilled_model_metric = metric[1]

        fig, axis = plt.subplots(5, 2)

        for i, ax in enumerate(axis.reshape(-1)[:-1]):
            ax.hist(ensemble_metric[:, i], bins=num_bins, alpha=0.5, density=True, label="Ensemble")
            ax.hist(distilled_model_metric[:, i], bins=num_bins, alpha=0.3, density=True, label="Distilled model")
            ax.set_xlabel("Logit " + metric_label + ", dim " + str(i+1))
            ax.legend()

        fig.suptitle(data_label + ", logit " + metric_label)
        plt.show()


def evaluate_uncertainty(data_set, data_label="", file_dir="../../dataloaders/data/distilled_model_predictions.h5",
                         teacher_ind=None):
    """Comparison of entropy histograms of ensemble and distilled model"""

    # Loading predictions from file
    with h5py.File(file_dir, 'r') as f:
        data_item = f[data_set]
        distilled_model_predictions = data_item["predictions"][()]
        ensemble_predictions = data_item["teacher-predictions"][()]

    if teacher_ind is not None:
        ensemble_predictions = ensemble_predictions[:, teacher_ind, :]

    distilled_model_tot_unc, distilled_model_ep_unc, distilled_model_al_unc = \
        metrics.uncertainty_separation_entropy(distilled_model_predictions, correct_nan=True)

    ensemble_tot_unc, ensemble_ep_unc, ensemble_al_unc = metrics.uncertainty_separation_entropy(ensemble_predictions,
                                                                                                correct_nan=True)

    metric_list = [[ensemble_tot_unc, distilled_model_tot_unc], [ensemble_ep_unc, distilled_model_ep_unc],
                   [ensemble_al_unc, distilled_model_al_unc]]
    metric_labels = ["total uncertainty", "epistemic uncertainty", "aleatoric uncertainty"]

    num_bins = 100

    fig, axis = plt.subplots(1, 3)

    for metric, metric_label, ax in zip(metric_list, metric_labels, axis.reshape(-1)):
        ensemble_metric = metric[0]
        distilled_model_metric = metric[1]

        ax.hist(ensemble_metric, bins=num_bins, alpha=0.5, density=True, label="Ensemble")
        ax.hist(distilled_model_metric, bins=num_bins, alpha=0.3, density=True, label="Distilled model")
        ax.set_xlabel(metric_label)
        ax.legend()

    fig.suptitle(data_label + ", " + "uncertainty")
    plt.show()


def check_distribution(data_set="train", file_dir="../../dataloaders/data/distilled_model_predictions.h5",
                       data_ind=None, teacher_ind=None):

    with h5py.File(file_dir, 'r') as f:
        data_item = f[data_set]
        mean = data_item["mean"][()]
        var = data_item["var"][()]
        samples = data_item["teacher-logits"][()]

    if data_ind is None:
        data_ind = np.random.choice(np.arange(0, samples.shape[0]), size=1)

    if teacher_ind is not None:
        samples = samples[:, teacher_ind, :]

    fig, axis = plt.subplots(5, 2)

    for i, ax in enumerate(axis.reshape(-1)[:-1]):

        x = np.linspace(mean[data_ind, i] - 3 * np.sqrt(var[data_ind, i]),
                        mean[data_ind, i] + 3 * np.sqrt(var[data_ind, i]), 100)
        ax.plot(x, scipy_stats.norm.pdf(x, mean[data_ind, i], np.sqrt(var[data_ind, i])))
        ax.plot(samples[data_ind, :, i], scipy_stats.norm.pdf(samples[data_ind, :, i],
                                                              mean[data_ind, i], np.sqrt(var[data_ind, i])), 'o')
        ax.set_xlabel("Dim " + str(i))

    fig.suptitle("Ensemble predictions and estimated distribution, {} data point".format(data_set))
    plt.show()


def evaluate_model(file_dir="../../dataloaders/data/distilled_model_predictions.h5", teacher_ind=None):

    data_list = ["train", "test"]
    data_labels = ["Train set", "Test set"]

    for data_set, data_label in zip(data_list, data_labels):
        evaluate_parameters(data_set, data_label=data_label, file_dir=file_dir, teacher_ind=teacher_ind)
        evaluate_uncertainty(data_set, data_label=data_label, file_dir=file_dir, teacher_ind=teacher_ind)


def create_index_permutation_file(max_val, file_dir):
    indices = np.random.choice(max_val, size=max_val, replace=False)
    np.save(file_dir, indices)


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))
    #evaluate_model(file_dir="../../dataloaders/data/distilled_model_predictions_ens10.h5",
    #               teacher_ind=[44, 8, 7, 2, 43, 47, 3, 15, 21, 24])
    #check_distribution(file_dir="../../dataloaders/data/distilled_model_predictions_ens10.h5")
    predictions(model_dir="../models/distilled_model_cifar10_vanilla_3",
                file_dir="../../dataloaders/data/distilled_model_vanilla_3_predictions.h5", vanilla=True)
    #predictions_corrupted_data(model_dir="../models/distilled_model_cifar10_1",
    #                           file_dir="../../dataloaders/data/distilled_model_corrupted_predictions_ens10.h5")

    #create_index_permutation_file(50, "data/ensemble_indices")
    #create_index_permutation_file(10000, "data/corrupted_data_indices")
    #create_index_permutation_file(50000, "data/training_data_indices")


if __name__ == "__main__":
    main()
