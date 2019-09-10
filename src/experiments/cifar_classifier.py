"""Classification experiment with CIFAR data"""
import logging
from pathlib import Path
from datetime import datetime
import torch
from src.distilled import dirichlet_CNN
from src.ensemble import ensemble
from src.ensemble import convolutional_classifier
import metrics
import utils
from src.dataloaders import cifar10

LOGGER = logging.getLogger(__name__)


def main():
    """Main"""

    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    LOGGER.info("Args: {}".format(args))
    device = utils.torch_settings(args.seed, args.gpu)

    cifar_data = cifar10.Cifar10Data()
    train_loader = torch.utils.data.DataLoader(cifar_data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=1)

    prob_ensemble = ensemble.Ensemble(cifar_data.num_classes)
    for _ in range(args.num_ensemble_members):
        model = convolutional_classifier.ConvolutionalClassifier(
            input_size=cifar_data.input_size,
            output_size=cifar_data.num_classes,
            device=device,
            learning_rate=args.lr)
        prob_ensemble.add_member(model)

    acc_metric = metrics.Metric(name="Acc", function=metrics.accuracy)
    prob_ensemble.add_metrics([acc_metric])
    prob_ensemble.train(train_loader, args.num_epochs)

    distilled_model = dirichlet_CNN.DirichletCNN(
        input_size=cifar_data.input_size,
        output_size=cifar_data.num_classes,
        teacher=prob_ensemble)
    distilled_model.train(train_loader, args.num_epochs * 2)


if __name__ == "__main__":
    main()
