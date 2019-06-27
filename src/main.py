""""Main entry point"""
from pathlib import Path
import numpy as np
import torch
from dataloaders import gaussian
import utils
import neural_network
import torchvision
import torchvision.transforms as transforms


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
                                               num_workers=2)
    # model = neural_network.NeuralNet(2, 3, 3, 2, lr=args.lr)
    model = neural_network.LinearNet(lr=args.lr)
    train(model, train_loader, args.num_epochs)
    for p in model.parameters():
        print("grad", p)


def main_dump():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)
    net = neural_network.DumpNet()
    net.train_full(trainloader)


def train(model, train_loader, num_epochs):
    for epoch in range(1, num_epochs + 1):
        loss = model.train_epoch(train_loader)
        print("Epoch {}: Loss: {}".format(epoch, loss))


if __name__ == "__main__":
    #main_dump()
    main()
