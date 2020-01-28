import src.dataloaders.cifar10_tensorflow as cifar10_tf
import numpy as np
import logging
import src.ensemble.tensorflow_ensemble as tensorflow_ensemble
import tensorflow as tf
import torch.nn as nn
import src.distilled.logits_teacher_from_file as logits_teacher_from_file
import src.ensemble.resnet20 as resnet20
import torch
import src.dataloaders.cifar10_ensemble_pred as cifar10_ensemble_pred
from pathlib import Path
from datetime import datetime
import src.utils as utils
import src.metrics as metrics
import src.dataloaders.cifar10_corrupted as cifar10_corrupted
import src.dataloaders.cifar10_benchmark_model_predictions as cifar10_benchmark_model_predictions
import src.dataloaders.cifar10 as cifar10
from matplotlib import pyplot as plt
import matplotlib.lines as matplot_lines
import src.ensemble.cifar_resnet as cifar_resnet
import h5py

LOGGER = logging.getLogger(__name__)


def ensemble_predictions():

    # Will load ensemble, make and save predictions here
    batch_size = 100
    train_data = cifar10_tf.Cifar10(batch_size=batch_size)
    test_data = cifar10_tf.Cifar10(batch_size=batch_size, train=False)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    models_dir = 'models/cifar-0508-ensemble_50/r'
    ensemble.load_ensemble(models_dir, num_members=50)

    get_ensemble_predictions_tf_dataset(ensemble, train_data, "cifar10_ensemble_train")
    get_ensemble_predictions_tf_dataset(ensemble, test_data, "cifar10_ensemble_test")


def ensemble_predictions_check():

    # TODO: Check acc/conf on corrupted data sets

    keys = ["train", "valid", "test"]

    # Predictions to compare with
    filepath = "../dataloaders/data/cifar_model_predictions.hdf5"
    with h5py.File(filepath, 'r') as f:
        model_item = f["ensemble"]

        for key in keys:
            data_set_item = model_item[key]
            probs = data_set_item["probs"][()]
            labels = data_set_item["labels"][()]

            predictions = np.argmax(probs, axis=-1)
            confidence = np.max(probs, axis=-1)
            acc = np.mean(predictions == labels)
            print("Predictions accuracy on {} data set: {}".format(key, acc))
            print("Predictions mean confidence on {} data set: {}".format(key, np.mean(confidence)))
            print("Predictions mean probs on {} data set; {}".format(key, np.mean(probs)))

    # Ensemble
    # training data (note: not sure that the split is the same
    train_set = cifar10_ensemble_pred.Cifar10Data()

    # validation data
    valid_set = cifar10_ensemble_pred.Cifar10Data(validation=True)

    # test data
    test_set = cifar10_ensemble_pred.Cifar10Data(train=False)

    data_sets = [train_set, valid_set, test_set]

    for key, data_set in zip(keys, data_sets):
        probs = data_set.set.data[1]
        ensemble_predictions = np.argmax(np.mean(probs, axis=1), axis=-1)
        acc = np.mean(ensemble_predictions == np.squeeze(data_set.set.targets))
        confidence = np.max(np.mean(probs, axis=1), axis=-1)

        #logits = data_set.set.data[2]
        #predictions = np.stack([scipy_softmax(x, axis=-1) for x in logits], axis=0)
        #np.save("../dataloaders/data/ensemble_predictions/ensemble_" + key + "_predictions", predictions)

        print("Ensemble accuracy on {} data set: {}".format(key, acc))
        print("Ensemble mean confidence on {} data set: {}".format(key, np.mean(confidence)))
        print("Ensemble mean probs on {} data set; {}".format(key, np.mean(probs)))

        for i in range(probs.shape[1]):
            acc = np.mean(np.argmax(probs[:, i, :], axis=-1) == np.squeeze(data_set.set.targets))
            print("Ensemble member accuracy on {} data set: {}".format(key, acc))


def ensemble_predictions_corrupted_data():

    # Load model
    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)

    models_dir = 'models/cifar-0508-ensemble_50/r'
    ensemble.load_ensemble(models_dir, num_members=50)

    # Load data, ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
    # "gaussian_noise", "glass_blur"
    corruption_list = ["impulse_noise", "motion_blur",
                       "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
                       "zoom_blur"]

    data_dir = "../dataloaders/data/ensemble_predictions/"
    file_dir = "../dataloaders/data"
    targets = []
    for i, corruption in enumerate(corruption_list):
        print(i)
        data = cifar10_corrupted.Cifar10Data(corruption=corruption, data_dir="../dataloaders/data/CIFAR-10-C/",
                                             torch=False)
        dataloader = torch.utils.data.DataLoader(data.set,
                                                 batch_size=100,
                                                 shuffle=False,
                                                 num_workers=0)

        predictions = []
        for batch in dataloader:
            inputs, labels = batch

            targets.append(labels)

            inputs = tf.convert_to_tensor(inputs.data.numpy())
            logits, preds = ensemble.predict(inputs)
            predictions.append(preds)

        predictions = tf.concat(predictions, axis=0)
        np.save(data_dir + corruption, predictions.numpy())

        targets = tf.concat(targets, axis=0)

        # Check accuracy
        new_preds = np.argmax(predictions, axis=-1)
        for j in np.arange(0, 5):
            saved_predictions = \
                cifar10_benchmark_model_predictions.Cifar10Data("ensemble", corruption, intensity=j, rep=0,  # Will just compare with one of the replications for now
                                                                filedir=file_dir)
            saved_preds = np.argmax(saved_predictions.set.predictions, axis=-1)
            saved_acc = np.mean(saved_preds == saved_predictions.set.labels)
            print("Previous prediction acc is {}".format(saved_acc))

            new_acc = np.mean(new_preds[j * 10000:(j + 1) * 10000] == targets[j * 10000:(j + 1) * 10000])
            print("New acc is {}".format(new_acc))

        if i == 0:
            np.save(data_dir + "labels", targets.numpy())


def get_ensemble_predictions_tf_dataset(ensemble, data, filepath):
    logits, predictions = [], []
    f = ensemble.predict  # tf.function(ensemble.predict)

    data.reset_batch()
    while data.next_batch:  # TODO: Fix this so it's a regular dataloader
        print(data.idx)
        inputs, labels = data.get_next_batch()
        logs, preds = f(inputs)
        logits.append(logs)
        predictions.append(preds)

    logits = tf.concat(logits, axis=0)
    predictions = tf.concat(predictions, axis=0)

    # Some kind of sanity check
    preds = np.argmax(np.mean(predictions.numpy(), axis=1), axis=-1)
    acc = np.mean(preds == np.squeeze(data.y))
    print("Accuracy is: {}".format(acc))

    np.save(filepath + "_logits", logits.numpy())
    np.save(filepath + "_predictions", predictions.numpy())

    return logits, predictions


def get_resnet_layers(softmax=False):
    # Adapted from benchmark article + extra info from https://towardsdatascience.com/resnets-for-cifar-10-e63e900524e0
    # conv2d defaults: stride=1, padding=0

    num_filters = 16

    layer_conv_1 = [nn.Conv2d(3, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01), nn.ReLU()]

    layer_1a = [nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01), nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01)]

    layer_1b = [nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01), nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01)]

    layer_1c = [nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01), nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01)]

    module_1 = [layer_1a, layer_1b, layer_1c]

    layer_2a_1 = [nn.Conv2d(num_filters, num_filters*2, 3, stride=2, padding=1), nn.BatchNorm2d(num_filters*2, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters*2, num_filters*2, 3, padding=1)]

    layer_2a_2 = [nn.Conv2d(num_filters, num_filters * 2, 3, stride=2, padding=1), nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01)]

    layer_2b_1 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1)]

    layer_2b_2 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1), nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01)]

    layer_2c_1 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1)]

    layer_2c_2 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1), nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01)]

    module_2 = [layer_2a_1, layer_2a_2,  layer_2b_1, layer_2b_2, layer_2c_1, layer_2c_2]

    layer_3a_1 = [nn.Conv2d(num_filters * 2, num_filters * 4, 3, stride=2, padding=1), nn.BatchNorm2d(num_filters*4, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1)]

    layer_3a_2 = [nn.Conv2d(num_filters * 2, num_filters * 4, 3, stride=2, padding=1), nn.BatchNorm2d(num_filters*4, eps=0.001, momentum=0.01)]

    layer_3b_1 = [nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 4, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1)]

    layer_3b_2 = [nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 4, eps=0.001, momentum=0.01)]

    layer_3c_1 = [nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 4, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1)]

    layer_3c_2 = [nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 4, eps=0.001, momentum=0.01)]

    module_3 = [layer_3a_1, layer_3a_2, layer_3b_1, layer_3b_2, layer_3c_1, layer_3c_2]

    layer_final = [nn.AvgPool2d(2), nn.Flatten(), nn.Linear(1024, 18)]

    module_list = [[layer_conv_1], module_1, module_2, module_3, [layer_final]]

    resnet_features = []
    for module in module_list:
        for block in module:
            resnet_features.append(nn.Sequential(*block))

    return resnet_features


def test_downloaded_resnet_network():

    # example output part of the model
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = cifar10.Cifar10Data(normalize=False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=2)

    resnet_model = cifar_resnet.ResNet(cifar_resnet.BasicBlock, [2, 2, 2, 2], learning_rate=args.lr * 0.5)

    acc_metric = metrics.Metric(name="Mean acc", function=metrics.accuracy)
    loss_metric = metrics.Metric(name="Mean loss", function=resnet_model.calculate_loss)
    resnet_model._add_metric(acc_metric)

    resnet_model.train(train_loader, num_epochs=12, reshape_targets=False)

    torch.save(resnet_model, "cifar10_resnet_dl")

    # Check accuracy on test data
    test_set = cifar10.Cifar10Data(train=False, normalize=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=2)


    counter = 0
    model_acc = 0
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(resnet_model.device)

        predicted_distribution = resnet_model.predict(inputs)
        model_acc += metrics.accuracy(predicted_distribution, labels.int())
        counter += 1

    model_acc = model_acc / counter
    print("Test accuracy: {}".format(model_acc))


def test_resnet_network():

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = cifar10.Cifar10Data(normalize=False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0)

    features_list = get_resnet_layers()

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    resnet20_model = resnet20.Resnet20(features_list, learning_rate=args.lr)

    acc_metric = metrics.Metric(name="Mean acc", function=metrics.accuracy)
    resnet20_model._add_metric(acc_metric)

    resnet20_model.train(train_loader, num_epochs=12, reshape_targets=False)

    torch.save(resnet20_model, "cifar10_resnet20")

    # Check accuracy on test data
    test_set = cifar10.Cifar10Data(train=False, normalize=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=2)


    counter = 0
    model_acc = 0
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(resnet20_model.device)

        predicted_distribution = resnet20_model.predict(inputs)
        model_acc += metrics.accuracy(predicted_distribution, labels.int())
        counter += 1

    model_acc = model_acc / counter
    print("Test accuracy: {}".format(model_acc))


def train_distilled_network():

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = cifar10_ensemble_pred.Cifar10Data()
    valid_set = cifar10_ensemble_pred.Cifar10Data(validation=True)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    # layer_list = [nn.Conv2d(3, 64, 3), nn.ReLU(), nn.MaxPool2d(2), torch.nn.BatchNorm2d(64),
    #               nn.Conv2d(64, 128, 3), nn.ReLU(), nn.MaxPool2d(2), torch.nn.BatchNorm2d(128),
    #               nn.Conv2d(128, 256, 5), nn.MaxPool2d(2), torch.nn.BatchNorm2d(256),
    #               nn.Flatten(), nn.Linear(2048, 128), nn.Linear(128, 18)]

    # layer_list = [nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
    #               nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten(),
    #               nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(),
    #               nn.Linear(84, 18)]
    #
    # features_list = nn.Sequential(*layer_list)

    features_list = get_resnet_layers()

    #data = train_set.set.data[0]

    # import torchvision
    # x = torch.stack([torchvision.transforms.ToTensor()(data[0, :, :, :])])
    # x = features(x)
    # assert x.shape == (1, 18)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    distilled_model = logits_teacher_from_file.LogitsTeacherFromFile(features_list,
                                                                     ensemble,
                                                                     learning_rate=args.lr * 0.1,
                                                                     scale_teacher_logits=True)

    loss_metric = metrics.Metric(name="Mean loss", function=distilled_model.calculate_loss)
    distilled_model.add_metric(loss_metric)

    distilled_dir = Path("models/distilled_model_cifar10")
    distilled_model.train(train_loader, num_epochs=100, validation_loader=valid_loader)
    torch.save(distilled_model, distilled_dir)


def check_trained_model():
    args = utils.parse_args()

    test_set = cifar10_ensemble_pred.Cifar10Data(train=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=0)

#    distilled_model = torch.load("models/distilled_model_cifar10")

    features_list = get_resnet_layers()
    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    distilled_model = logits_teacher_from_file.LogitsTeacherFromFile(features_list,
                                                                     ensemble,
                                                                     learning_rate=args.lr * 0.1,
                                                                     scale_teacher_logits=True)

    file_dir = "models/model_data/"
    distilled_model.load_weights(file_dir)

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
    print(distilled_model_acc)
    LOGGER.info("Distilled model accuracy on {} data {}".format("test", distilled_model_acc))


def predictions_on_corrupted_data():

    # Load data
    corruption_list = ["brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
                       "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
                       "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise",
                       "zoom_blur"]

    for corruption in corruption_list:
        data = cifar10_corrupted.Cifar10Data(corruption=corruption)
        loader = torch.utils.data.DataLoader(data.set,
                                             batch_size=100,
                                             shuffle=False,
                                             num_workers=0)


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
    # Får eventuellt ladda alla modeller och göra test-prediktioner
    # Jag har ju redan för ensemblen
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
                data = cifar10_benchmark_model_predictions.Cifar10Data(model=model, corruption=corruption,
                                                                       intensity=intensity, filedir=filedir)

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


if __name__ == "__main__":
    #train_distilled_network()
    #check_trained_model()
    #repeat_acc_ece_exp()
    ensemble_predictions_corrupted_data()
    #ensemble_predictions_check()
    #test_resnet_network()
    #test_downloaded_resnet_network()
