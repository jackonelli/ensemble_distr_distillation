"""Test of uncertainty in simple nll regression ensemble"""

from pathlib import Path
from datetime import datetime
import logging
import torch
import sys
print(sys.path)

from src.dataloaders import noisy_one_dim_regression
import src.utils as utils
from src.ensemble import ensemble
from src.ensemble import simple_regressor
import src.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def make_plots(train_loader, test_loader, prob_ensemble):

    test_data = test_loader.dataset.get_full_data()
    test_x = test_data[:, 0][:, np.newaxis]
    test_y = test_data[:, 1]

    y_hat = prob_ensemble.predict(torch.tensor(test_x, dtype=torch.float32))
    total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = \
        metrics.uncertainty_separation_variance(y_hat, None)

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].plot(test_x, test_y, '.')
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('y')
    ax[0, 0].legend(['Test data'])

    for i in np.arange(prob_ensemble.num_samples):
        ax[0, 1].plot(test_x, y_hat[:, :, i], 'b.', markersize=1.5)
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('$\hat{y}$')
    ax[0, 1].legend(['Predictive distribution'])

    train_data = train_loader.dataset.get_full_data()
    ax[0, 2].hist(train_data[:, 0])
    ax[0, 2].set_xlabel('x')
    ax[0, 2].set_ylabel('\u03C1')
    ax[0, 2].legend(['Training data density'])

    ax[1, 0].plot(test_x, total_uncertainty, '.')
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('var(y)')
    ax[1, 0].legend(['Total uncertainty'])

    ax[1, 1].plot(test_x, epistemic_uncertainty, '.')
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('var(y)-var(y|\u03B8)')
    ax[1, 1].legend(['Epistemic uncertainty'])

    ax[1, 2].plot(test_x, aleatoric_uncertainty, '.')
    ax[1, 2].set_xlabel('x')
    ax[1, 2].set_ylabel('var(y|\u03B8)')
    ax[1, 2].legend(['Aleatoric uncertainty'])

    plt.subplots_adjust(bottom=0.1, left=0.05, right=0.99, top=0.95)
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
    train_data = noisy_one_dim_regression.SyntheticRegressionData(store_file=Path("data/one_dim_reg_1000"))

    valid_data = noisy_one_dim_regression.SyntheticRegressionData(store_file=Path("data/one_dim_reg_valid_500"),
                                                                  n_samples=500)

    test_data = noisy_one_dim_regression.SyntheticRegressionData(store_file=Path("data/one_dim_reg_test_500"),
                                                                 train=False, n_samples=500)

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
    hidden_size = 10
    ensemble_output_size = 2

    prob_ensemble = ensemble.Ensemble(ensemble_output_size)
    ensemble_filepath = Path("models/simple_reg_ensemble_10")

    for _ in range(2):
        model = simple_regressor.SimpleRegressor(input_size,
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
