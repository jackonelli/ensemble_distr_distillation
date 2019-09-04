""""Main entry point"""
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import torch
from dataloaders import gaussian
import utils
import models
from distilled import dirichlet_probability_distribution
from ensemble import ensemble
import metrics

LOGGER = logging.getLogger(__name__)


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
    output_size = 2

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=1)
    prob_ensemble = ensemble.Ensemble(output_size)
    for _ in range(args.num_ensemble_members):
        model = models.NeuralNet(input_size,
                                 3,
                                 3,
                                 output_size,
                                 device=device,
                                 learning_rate=args.lr)
        prob_ensemble.add_member(model)
    acc_metric = metrics.Metric(name="Acc", function=metrics.accuracy)
    prob_ensemble.add_metrics([acc_metric])
    prob_ensemble.train(train_loader, args.num_epochs)

    distilled_model = dirichlet_probability_distribution.DirichletProbabilityDistribution(
        input_size,
        3,
        3,
        output_size,
        teacher=prob_ensemble,
        device=device,
        learning_rate=args.lr * 0.1)
    distilled_model.train(train_loader, args.num_epochs * 2)


if __name__ == "__main__":
    main()
