""""Main entry point"""
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import torch
import sys
print(sys.path)

from src.dataloaders import gaussian
import src.utils as utils
from src.distilled import logits_probability_distribution
from src.ensemble import ensemble
from src.ensemble import simple_classifier
import src.metrics as metrics
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def make_plots(distilled_model):
    data = gaussian.SyntheticGaussianData(n_samples=400, train=False, store_file=Path("data/one_dim_reg_500"))

    test_loader = torch.utils.data.DataLoader(data,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=0)

    predictions = np.zeros((data.n_samples, distilled_model.output_size))
    all_x = np.zeros((data.n_samples, 1))
    all_y = np.zeros((data.n_samples, 1))

    idx = 0
    for batch in test_loader:
        inputs, targets = batch

        predictions[idx * test_loader.batch_size:(idx + 1) * test_loader.batch_size, :, :] = \
            distilled_model.predict(inputs, t=None).data.numpy()

        all_x[idx * test_loader.batch_size:(idx + 1) * test_loader.batch_size, :] = inputs
        all_y[idx * test_loader.batch_size:(idx + 1) * test_loader.batch_size, :] = targets

        idx += 1

    plt.scatter(np.squeeze(all_x), np.squeeze(all_y), label="Data", marker='.')

    plt.errorbar(np.squeeze(all_x), predictions[:, 0],
                 np.sqrt(predictions[:, 1]),
                 label="Distilled model predictions", marker='.', ls='none')

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
    data = gaussian.SyntheticGaussianData(
        mean_0=[0, 0],
        mean_1=[-3, -3],
        cov_0=np.eye(2),
        cov_1=np.eye(2),
        store_file=Path("data/2d_gaussian_1000"))

    # TODO: Automated dims
    input_size = 2
    hidden_size = 3
    ensemble_output_size = 2

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=0)

    prob_ensemble = ensemble.Ensemble(ensemble_output_size)
    # for _ in range(args.num_ensemble_members):
    #     model = simple_classifier.SimpleClassifier(input_size,
    #                                                hidden_size,
    #                                                hidden_size,
    #                                                output_size,
    #                                                device=device,
    #                                                learning_rate=args.lr)
    #     prob_ensemble.add_member(model)
    # acc_metric = metrics.Metric(name="Acc", function=metrics.accuracy)
    # prob_ensemble.add_metrics([acc_metric])
    # prob_ensemble.train(train_loader, args.num_epochs)

    ensemble_filepath = Path("models/simple_class_logits_ensemble")

    #  prob_ensemble.save_ensemble(ensemble_filepath)
    prob_ensemble.load_ensemble(ensemble_filepath)

    distilled_output_size = 4
    distilled_model = logits_probability_distribution.LogitsProbabilityDistribution(
        input_size,
        hidden_size,
        hidden_size,
        distilled_output_size,
        teacher=prob_ensemble,
        device=device,
        learning_rate=args.lr * 0.1)
    distilled_model.train(train_loader, args.num_epochs * 2)


if __name__ == "__main__":
    main()
