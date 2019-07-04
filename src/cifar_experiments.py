from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import torch
from dataloaders import cifar10
import utils
import models
import distilled_network
import ensemble
import experiments
from new_models import cifar_net

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
    data = cifar10.Cifar10Data()
    train_loader = torch.utils.data.DataLoader(data.set,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=1)
    model = cifar_net.EnsembleNet(device=device, learning_rate=args.lr)
    prob_ensemble = ensemble.Ensemble()
    prob_ensemble.add_member(model)
    prob_ensemble.train(train_loader, args.num_epochs)

    distilled = cifar_net.DistilledNet(prob_ensemble,
                                       device=device,
                                       learning_rate=args.lr)
    distilled.train(train_loader, args.num_epochs)


if __name__ == "__main__":
    main()
