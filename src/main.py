""""Main entry point"""
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import torch
from dataloaders import gaussian
import utils
import models
import distilled_network
import ensemble
import experiments

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
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=1)
    model = models.NeuralNet(2, 3, 3, 2, device=device, learning_rate=args.lr)
    prob_ensemble = ensemble.Ensemble()
    prob_ensemble.add_member(model)
    prob_ensemble.train(train_loader, args.num_epochs)

    distilled_model = distilled_network.PlainProbabilityDistribution(
        2, 3, 3, 2, model, device=device, lr=args.lr)
    distilled_model.train(train_loader, args.num_epochs)


if __name__ == "__main__":
    main()
