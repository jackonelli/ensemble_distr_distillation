""""Main entry point"""
from pathlib import Path
import numpy as np
import torch
from dataloaders import gaussian
import utils
import neural_network


def main():
    """Main"""
    args = utils.parse_args()
    device = utils.torch_settings(args.seed)
    data = gaussian.SyntheticGaussianData(
        mean_0=[0, 0],
        mean_1=[1, 0],
        cov_0=np.eye(2),
        cov_1=np.eye(2),
        store_file=Path("data/2d_gaussian_1000"))
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=2)
    model = neural_network.NeuralNet(2, 10, 5, 2)
    train(model, train_loader, args.num_epochs)


def train(model, train_loader, num_epochs):
    for epoch in range(1, num_epochs + 1):
        loss = model.train_epoch(train_loader)
        print("Epoch {}: Loss: {}".format(epoch, loss))


if __name__ == "__main__":
    main()
