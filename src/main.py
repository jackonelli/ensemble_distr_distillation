""""Main entry point"""
from pathlib import Path
import numpy as np
import torch
from dataloaders import gaussian
import utils
import models
import distilled_network


def main():
    """Main"""
    args = utils.parse_args()
    device = utils.torch_settings(args.seed)
    data = gaussian.SyntheticGaussianData(
        mean_0=[0, 0],
        mean_1=[-3, -3],
        cov_0=np.eye(2),
        cov_1=np.eye(2),
        store_file=Path("data/2d_gaussian_1000"))
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=0)
    model = models.NeuralNet(2, 3, 3, 2, lr=args.lr)
    # model = models.LinearNet(lr=args.lr)
    model.train(train_loader, args.num_epochs)
    for p in model.parameters():
        print("grad", p)

    distilled_model = distilled_network.NeuralNet(2,
                                                  3,
                                                  3,
                                                  2,
                                                  model,
                                                  lr=args.lr)
    distilled_model.train(train_loader, args.num_epochs)


if __name__ == "__main__":
    main()
