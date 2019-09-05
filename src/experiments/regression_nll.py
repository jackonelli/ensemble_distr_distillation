from pathlib import Path
from datetime import datetime
import logging
import torch
import sys
print(sys.path)

from dataloaders import one_dim_regression
import utils
from distilled import dirichlet_probability_distribution
from ensemble import ensemble
from ensemble import simple_regressor
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

    data = one_dim_regression.SyntheticRegressionData(
            store_file=Path("data/one_dim_reg_1000"))

    input_size = 1
    hidden_size = 10
    output_size = 2

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=1)

    prob_ensemble = ensemble.Ensemble(output_size)
    for _ in range(args.num_ensemble_members):
        model = simple_regressor.SimpleRegressor(input_size,
                                                 hidden_size,
                                                 hidden_size,
                                                 output_size,
                                                 device=device,
                                                 learning_rate=args.lr)
        prob_ensemble.add_member(model)

    err_metric = metrics.Metric(name="Acc", function=metrics.accuracy)
    prob_ensemble.add_metrics([err_metric])
    prob_ensemble.train(train_loader, args.num_epochs)

    # distilled_model = dirichlet_probability_distribution.DirichletProbabilityDistribution(
    #     input_size,
    #     hidden_size,
    #     hidden_size,
    #     output_size,
    #     teacher=prob_ensemble,
    #     device=device,
    #     learning_rate=args.lr * 0.1)
    # distilled_model.train(train_loader, args.num_epochs * 2)


if __name__ == "__main__":
    main()
