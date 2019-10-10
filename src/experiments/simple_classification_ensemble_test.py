"""Test of uncertainty in simple ensemble classifier"""

from pathlib import Path
from datetime import datetime
import logging
import torch
import sys
print(sys.path)

from src.dataloaders import noisy_gaussian
import src.utils as utils
from src.ensemble import ensemble
from src.ensemble import simple_classifier
import src.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def make_plots(train_loader, test_loader, model):

    test_data = test_loader.dataset.get_full_data()
    test_x = test_data[:, 0]
    test_y = test_data[:, 1]

    p_hat = np.stack(utils.predict(test_x, model), axis=-1)

    total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = \
        utils.get_uncertainty_separation_entropy(test_x, model)

    label_1_inds = (test_y[:, 0] == 1)
    label_2_inds = (test_y[:, 1] == 1)
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].plot(test_x[label_1_inds], test_y[label_1_inds, 0] * 0 + 1, '.', color='b')
    ax[0, 0].plot(test_x[label_2_inds], test_y[label_2_inds, 0] * 0 + 1, '.', color='r')
    ax[0, 0].legend(['Class 1', 'Class 2'])
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('y')
    ax[0, 0].legend(['Test data'])

    for i in np.arange(p_hat.shape[-1]):
        ax[0, 1].plot(test_x, p_hat[:, 0, i], '.', color='b')
        ax[0, 1].plot(test_x, p_hat[:, 1, i], '.', color='r')

    ax[0, 1].legend(['Class 1', 'Class 2'])
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('$\hat{y}$')
    ax[0, 1].legend(['Class 1', 'Class 2'])

    train_data = train_loader.dataset.get_full_data()
    train_x = train_data[:, 0]
    train_y = train_data[:, 1]
    label_1_inds_train = (train_y[:, 0] == 1)
    label_2_inds_train = (train_y[:, 1] == 1)
    ax[0, 2].hist(train_x[label_1_inds_train], color='b')
    ax[0, 2].hist(train_x[label_2_inds_train], color='r')
    ax[0, 2].set_xlabel('x')
    ax[0, 2].set_ylabel('\u03C1')
    ax[0, 2].legend(['Class 1, training data density', 'Class 2, training data density'])

    ax[1, 0].plot(test_x, total_uncertainty, '.')
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('H(y)')
    ax[1, 0].legend(['Total uncertainty'], loc='upper right')

    ax[1, 1].plot(test_x, epistemic_uncertainty, '.')
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('H(y)-H(y|\u03B8)')
    ax[1, 1].legend(['Epistemic uncertainty'], loc='upper right')

    ax[1, 2].plot(test_x, aleatoric_uncertainty, '.')
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('H(y|\u03B8)]')
    ax[1, 2].legend(['Aleatoric uncertainty'], loc='upper right')

    plt.subplots_adjust(bottom=0.1, left=0.05, right=1.0, top=0.95)

    plt.show()


def main():
    """Main"""
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))
    device = utils.torch_settings(args.seed, args.gpu)
    LOGGER.info("Creating dataloader")

    # NOTE: MAYBE SHOULD NOT RESAMPLE DATA

    mean_0 = -3
    mean_1 = 3
    cov_0 = 1
    cov_1 = 1

    train_data = noisy_gaussian.SyntheticRegressionData(mean_0, mean_1, cov_0, cov_1,
                                                        store_file=Path("data/one_dim_class_1000"))

    valid_data = noisy_gaussian.SyntheticRegressionData(mean_0, mean_1, cov_0, cov_1,
                                                        store_file=Path("data/one_dim_class_1000"),
                                                        n_samples=500)

    test_data = noisy_gaussian.SyntheticRegressionData(mean_0, mean_1, cov_0, cov_1,
                                                       store_file=Path("data/one_dim_class_test_500"), train=False,
                                                       n_samples=500)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=0)

    input_size = 1
    hidden_size = 5
    ensemble_output_size = 2

    prob_ensemble = ensemble.Ensemble(ensemble_output_size)
    ensemble_filepath = Path("models/simple_class_ensemble_10")

    for _ in range(args.num_ensemble_members):
        model = simple_classifier.SimpleClassifier(input_size,
                                                   hidden_size,
                                                   hidden_size,
                                                   ensemble_output_size,
                                                   device=device,
                                                   learning_rate=args.lr)
        prob_ensemble.add_member(model)

    err_metric = metrics.Metric(name="Err", function=metrics.squared_error)
    prob_ensemble.add_metrics([err_metric])
    prob_ensemble.train(train_loader, args.num_epochs, valid_loader)

    prob_ensemble.save_ensemble(ensemble_filepath)

    #prob_ensemble.load_ensemble(ensemble_filepath)

    prob_ensemble.calc_metrics(test_loader)

    make_plots(train_loader, test_loader, prob_ensemble)


if __name__ == "__main__":
    main()
