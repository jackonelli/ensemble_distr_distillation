import numpy as np
import logging
import torch
import src.dataloaders.cifar10_benchmark_model_predictions as cifar10_benchmark_model_predictions
import matplotlib.lines as matplot_lines
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime
from src import utils


LOGGER = logging.getLogger(__name__)

# SNYGGARE OM JAG UTNYTTJAR ATT VI HAR FUNKTIONER I METRICS?
def calc_acc(data_loader):
    # Return accuracy over data loader

    acc = 0
    idx = 0
    for batch in data_loader:
        predictions, labels = batch
        predicted_labels = np.argmax(predictions, axis=-1)
        acc += np.mean(predicted_labels.data.numpy() == labels.data.numpy())
        idx += 1

    return acc / idx


def calc_ece(data_loader):
    # Return ece over data loader
    # split_values should indicate the upper bounds on the confidence of the buckets

    predictions = data_loader.dataset.predictions
    labels = data_loader.dataset.labels
    num_samples = labels.shape[0]

    confidence = np.max(predictions, axis=-1)
    predicted_labels = np.argmax(predictions, axis=-1)

    # Sort the data
    ind = np.argsort(confidence)

    confidence = confidence[ind]
    predicted_labels = predicted_labels[ind]
    labels = labels[ind]

    # Will go for quartiles
    split_values = np.quantile(confidence, q=[0.25, 0.50, 0.75, 1.0], axis=0).tolist()
    split_values = np.array(split_values)

    num_buckets = split_values.shape[0]
    acc = np.zeros((num_buckets, 1))
    conf = np.zeros((num_buckets, 1))
    bucket_count = np.zeros((num_buckets, 1))

    j = 0
    for i in range(num_buckets):
        while confidence[j] <= split_values[i]:
            acc[i] += predicted_labels[j] == labels[j]
            conf[i] += confidence[j]
            bucket_count[i] += 1
            j += 1

            if j >= confidence.shape[0]:
                break

    acc = acc / bucket_count
    conf = conf / bucket_count

    ece = np.sum((bucket_count / num_samples) * np.abs(acc - conf))

    return ece


def make_boxplot(data_list, label="ACC", model_list=None, colors=None, max_y=1.0):
    """Make boxplot over data in data_list
    M = number of models, N = number of data_sets, I = number of intensities
    data_list: list of length I with matrices of size (N, M)
    """

    num_intensities = len(data_list)
    xlab = np.arange(0, num_intensities)
    fig, axis = plt.subplots(2, int(np.ceil(num_intensities/2)))

    for i, (data, ax) in enumerate(zip(data_list, axis.reshape(-1)[0:num_intensities])):

        bplot = ax.boxplot(data, patch_artist=True)
        ax.set_ylabel(label)
        ax.set_title("Intensity " + str(xlab[i]))

        if colors is not None:
            assert len(colors) == data.shape[-1]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        plt.setp(bplot['medians'], color='black')

        if colors is not None:
            custom_lines = []
            for j in range(len(colors)):
                custom_lines.append(matplot_lines.Line2D([0], [0], color=colors[j], lw=2))

        if model_list is not None:
            assert len(model_list) == len(custom_lines)
            ax.legend(custom_lines, model_list)

           # if colors is not None:
           #     for j in range(len(colors)):
           #         leg.legendHandles[j].set_color(colors[j])

        ax.set_ylim([0.0, max_y])

    plt.show()


def make_test_predictions():
    LOGGER.info("Hallå där")
    # Får eventuellt ladda alla modeller och göra test-prediktioner
    # Jag har ju redan för ensemblen
    # Va, har jag inte detta?
    pass


def repeat_acc_ece_exp():

    filedir = "../dataloaders/data/"

    model_list = ["vanilla", "temp_scaling", "ensemble", "dropout", "ll_dropout", "svi", "ll_svi"]
    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                       "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "pixelate",
                       "saturate", "shot_noise", "spatter", "speckle_noise", "zoom_blur"]
    intensity_list = [0, 1, 2, 3, 4, 5]

    acc_list = []
    ece_list = []
    for intensity in intensity_list:
        acc = np.zeros((len(corruption_list), len(model_list)))
        ece = np.zeros((len(corruption_list), len(model_list)))
        for i, corruption in enumerate(corruption_list):
            for j, model in enumerate(model_list):
                data = cifar10_benchmark_model_predictions.Cifar10DataPredictions(model=model, corruption=corruption,
                                                                                  intensity=intensity, data_dir=filedir)

                data_loader = torch.utils.data.DataLoader(data.set,
                                                          batch_size=100,
                                                          shuffle=False,
                                                          num_workers=0)

                acc[i, j] = calc_acc(data_loader)
                ece[i, j] = calc_ece(data_loader)

        acc_list.append(acc)
        ece_list.append(ece)

    model_list_text = ["Vanilla", "Temp Scaling", "Ensemble", "Dropout", "LL Dropout", "SVI", "LL SVI"]
    colors = ['gray', 'r', 'g', 'b', 'c', 'm', 'y']
    make_boxplot(acc_list, model_list=model_list_text, colors=colors)
    make_boxplot(ece_list, label="ECE", model_list=model_list_text,
                 colors=colors, max_y=0.4)


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)


if __name__ == "__main__":
    #repeat_acc_ece_exp()
    main()
