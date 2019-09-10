from pathlib import Path
from datetime import datetime
import logging
import torch
import sys
print(sys.path)

from src.dataloaders import one_dim_regression
import src.utils as utils
from src.distilled import niw_probability_distribution
from src.ensemble import ensemble
from src.ensemble import simple_regressor
import src.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def plot_predictions(prob_ensemble):

    data = one_dim_regression.SyntheticRegressionData(n_samples=500, train=False,
                                                      store_file=Path("data/one_dim_reg_500"))

    test_loader = torch.utils.data.DataLoader(data,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=0)

    x_batch, y_batch = next(iter(test_loader))

    predictions = prob_ensemble.predict(x_batch, t=None)

    plt.scatter(np.squeeze(x_batch), y_batch, label="Data")

    for i in np.arange(prob_ensemble.size):
        plt.errorbar(np.squeeze(x_batch), predictions[:, i, 0].data.numpy(),
                     predictions[:, i, 1].data.numpy(),
                     label="Ensemble member predictions " + str(i), marker='.')

    plt.legend()
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

    data = one_dim_regression.SyntheticRegressionData(
            store_file=Path("data/one_dim_reg_1000"))

    input_size = 1
    hidden_size = 5
    ensemble_output_size = 2

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=16,
                                               shuffle=True,
                                               num_workers=0)

    prob_ensemble = ensemble.Ensemble(ensemble_output_size)
    # for _ in range(args.num_ensemble_members):
    #     model = simple_regressor.SimpleRegressor(input_size,
    #                                              hidden_size,
    #                                              hidden_size,
    #                                              output_size,
    #                                              device=device,
    #                                              learning_rate=args.lr)
    #     prob_ensemble.add_member(model)
    #
    # err_metric = metrics.Metric(name="Err", function=metrics.squared_error)
    # prob_ensemble.add_metrics([err_metric])
    # prob_ensemble.train(train_loader, args.num_epochs)

    ensemble_filepath = Path("models/simple_reg_ensemble")
    prob_ensemble.load_ensemble(ensemble_filepath)

    # plot_predictions(prob_ensemble)

    #prob_ensemble.save_ensemble(ensemble_filepath)

    distilled_output_size = 4

    distilled_model = niw_probability_distribution.NiwProbabilityDistribution(
        input_size,
        hidden_size,
        hidden_size,
        distilled_output_size,
        target_dim=1,
        teacher=prob_ensemble,
        device=device,
        learning_rate=args.lr * 0.1)
    distilled_model.train(train_loader, args.num_epochs * 2)


if __name__ == "__main__":
    main()
