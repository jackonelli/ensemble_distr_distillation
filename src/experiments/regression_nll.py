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

    data = one_dim_regression.SyntheticRegressionData(n_samples=400, train=False,
                                                      store_file=Path("data/one_dim_reg_500"))

    test_loader = torch.utils.data.DataLoader(data,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=0)

    predictions = np.zeros((data.n_samples, prob_ensemble.size, prob_ensemble.output_size))
    all_x = np.zeros((data.n_samples, 1))
    all_y = np.zeros((data.n_samples, 1))

    idx = 0
    for batch in test_loader:
        inputs, targets = batch

        predictions[idx*test_loader.batch_size:(idx + 1)*test_loader.batch_size, :, :] = \
            prob_ensemble.predict(inputs, t=None).data.numpy()

        all_x[idx * test_loader.batch_size:(idx + 1) * test_loader.batch_size, :] = inputs
        all_y[idx * test_loader.batch_size:(idx + 1) * test_loader.batch_size, :] = targets

        idx += 1

    plt.scatter(np.squeeze(all_x), np.squeeze(all_y), label="Data", marker='.')

    for i in np.arange(prob_ensemble.size):
        plt.errorbar(np.squeeze(all_x), predictions[:, i, 0],
                     np.sqrt(predictions[:, i, 1]),
                     label="Ensemble member " + str(i+1) + " predictions", marker='.', ls='none')

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

    # NOTE: MAYBE SHOULD NOT RESAMPLE DATA
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
    ensemble_filepath = Path("models/simple_reg_ensemble")
    #
    # for _ in range(args.num_ensemble_members):
    #     model = simple_regressor.SimpleRegressor(input_size,
    #                                              hidden_size,
    #                                              hidden_size,
    #                                              ensemble_output_size,
    #                                              device=device,
    #                                              learning_rate=args.lr)
    #     prob_ensemble.add_member(model)
    #
    # err_metric = metrics.Metric(name="Err", function=metrics.squared_error)
    # prob_ensemble.add_metrics([err_metric])
    # prob_ensemble.train(train_loader, args.num_epochs)
    #
    # prob_ensemble.save_ensemble(ensemble_filepath)

    prob_ensemble.load_ensemble(ensemble_filepath)

    plot_predictions(prob_ensemble)

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
