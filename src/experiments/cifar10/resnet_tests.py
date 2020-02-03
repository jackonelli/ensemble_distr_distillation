import numpy as np
import logging
import torch
import torch.nn as nn
from src import utils
from src import metrics
from src.dataloaders import cifar10
from src.ensemble import cifar_resnet
from src.ensemble import resnet20

from pathlib import Path
from datetime import datetime


LOGGER = logging.getLogger(__name__)


def get_resnet_layers():
    # Adapted from benchmark article + extra info from https://towardsdatascience.com/resnets-for-cifar-10-e63e900524e0
    # conv2d defaults: stride=1, padding=0

    num_filters = 16

    layer_conv_1 = [nn.Conv2d(3, num_filters, 3, padding=1), nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01),
                    nn.ReLU()]

    layer_1a = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01), nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01)]

    layer_1b = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01), nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01)]

    layer_1c = [nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01), nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 3, padding=1),
                nn.BatchNorm2d(num_filters, eps=0.001, momentum=0.01)]

    module_1 = [layer_1a, layer_1b, layer_1c]

    layer_2a_1 = [nn.Conv2d(num_filters, num_filters * 2, 3, stride=2, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1)]

    layer_2a_2 = [nn.Conv2d(num_filters, num_filters * 2, 3, stride=2, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01)]

    layer_2b_1 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1)]

    layer_2b_2 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01)]

    layer_2c_1 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1)]

    layer_2c_2 = [nn.Conv2d(num_filters * 2, num_filters * 2, 3, padding=1),
                  nn.BatchNorm2d(num_filters * 2, eps=0.001, momentum=0.01)]

    module_2 = [layer_2a_1, layer_2a_2, layer_2b_1, layer_2b_2, layer_2c_1, layer_2c_2]

    layer_3a_1 = [nn.Conv2d(num_filters * 2, num_filters * 4, 3, stride=2, padding=1),
                  nn.BatchNorm2d(num_filters * 4, eps=0.001, momentum=0.01), nn.ReLU(),
                  nn.Conv2d(num_filters * 4, num_filters * 4, 3, padding=1)]

    layer_3a_2 = [nn.Conv2d(num_filters * 2, num_filters * 4, 3, stride=2, padding=1),
                  nn.BatchNorm2d(num_filters * 4, eps=0.001, momentum=0.01)]

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


def test_resnet_network():
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    N = 50000
    train_ind = np.random.choice(np.arange(0, N), size=40000, replace=False)
    ind = np.stack([i for i in np.arange(N) if i not in train_ind])
    valid_ind = ind

    train_set = cifar10.Cifar10Data(ind=train_ind, augmentation=True)
    valid_set = cifar10.Cifar10Data(ind=valid_ind)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=2)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=2)

    test_set = cifar10.Cifar10Data(train=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=2)

    features_list = get_resnet_layers()

    resnet20_model = resnet20.Resnet20(features_list, learning_rate=args.lr)

    acc_metric = metrics.Metric(name="Mean acc", function=metrics.accuracy)
    loss_metric = metrics.Metric(name="Mean loss", function=resnet20_model.calculate_loss)
    resnet20_model._add_metric(acc_metric)
    resnet20_model._add_metric(loss_metric)

    resnet20_model.train(train_loader, validation_loader=valid_loader, num_epochs=200, reshape_targets=False)

    # Check accuracy on test data
    resnet20_model.eval_mode()

    counter = 0
    model_acc = 0
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(resnet20_model.device)

        predicted_distribution = resnet20_model.predict(inputs)
        model_acc += metrics.accuracy(predicted_distribution.to(torch.device("cpu")), labels.int())
        counter += 1

    model_acc = model_acc / counter
    LOGGER.info("Test accuracy: {}".format(model_acc))
    resnet20_model.to(torch.device('cpu'))
    torch.save(resnet20_model.state_dict(), "models/model_cifar10_resnet20")


def test_downloaded_resnet_network():
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    N = 50000
    train_ind = np.random.choice(np.arange(0, N), size=40000, replace=False)
    ind = np.stack([i for i in np.arange(N) if i not in train_ind])
    valid_ind = ind

    train_set = cifar10.Cifar10Data(ind=train_ind, augmentation=True)
    valid_set = cifar10.Cifar10Data(ind=valid_ind)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=2)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=2)

    test_set = cifar10.Cifar10Data(train=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=2)

    resnet_model = cifar_resnet.ResNet(cifar_resnet.Bottleneck, [2, 2, 2, 2], learning_rate=args.lr * 0.1)

    acc_metric = metrics.Metric(name="Mean acc", function=metrics.accuracy)
    loss_metric = metrics.Metric(name="Mean loss", function=resnet_model.calculate_loss)
    resnet_model._add_metric(acc_metric)
    resnet_model._add_metric(loss_metric)

    resnet_model.train(train_loader, validation_loader=valid_loader, num_epochs=200, reshape_targets=False)
    # Check accuracy on test data

    resnet_model.eval_mode()
    counter = 0
    model_acc = 0

    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(resnet_model.device)

        predicted_distribution = resnet_model.predict(inputs)
        model_acc += metrics.accuracy(predicted_distribution.to(torch.device("cpu")), labels.int())
        counter += 1

    model_acc = model_acc / counter
    LOGGER.info("Test accuracy: {}".format(model_acc))

    torch.save(resnet_model.state_dict(), "models/model_cifar10_resnet_dl")


def main():
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))
    test_downloaded_resnet_network()


if __name__ == "__main__":
    main()
