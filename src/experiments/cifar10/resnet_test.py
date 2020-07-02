import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

from src import utils
from src import metrics
from src.dataloaders import cifar10
from src.ensemble import cifar_resnet
from src.experiments.cifar10 import resnet_utils

LOGGER = logging.getLogger(__name__)


def test_downloaded_resnet_network():
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

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

    test_set = cifar10.Cifar10Data(train=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=0)

    resnet_model = cifar_resnet.ResNet(resnet_utils.Bottleneck, [2, 2, 2, 2], learning_rate=args.lr)

    resnet_model.train(train_loader, validation_loader=valid_loader, num_epochs=args.num_epochs, reshape_targets=False)
    # Check accuracy on test data

    resnet_model.eval_mode()
    counter = 0
    model_acc = 0

    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(resnet_model.device)

        predicted_distribution = resnet_model.predict(inputs)
        model_acc += metrics.accuracy(predicted_distribution.to(torch.device("cpu")), labels.long())
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
