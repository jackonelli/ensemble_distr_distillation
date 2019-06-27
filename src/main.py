""""Main entry point"""
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as torch_transforms
from dataloaders import gaussian
import utils
import models
import extra_models
import torchvision


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
    # model = models.NeuralNet(2, 3, 3, 2, lr=args.lr)
    model = models.LinearNet(lr=args.lr)
    model.train(train_loader, args.num_epochs)
    for p in model.parameters():
        print("grad", p)


def main_dump():
    transform = torch_transforms.Compose([
        torch_transforms.ToTensor(),
        torch_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)
    net = extra_models.DumpNet()
    net.train_full(trainloader)


if __name__ == "__main__":
    #main_dump()
    main()
