""""Main entry point"""
import argparse
from pathlib import Path
import numpy as np
from dataloaders import gaussian


def main():
    """Main"""
    args = parse_args()
    gaussian.SyntheticGaussianData(mean_0=[0, 0],
                                   mean_1=[1, 0],
                                   cov_0=np.eye(2),
                                   cov_1=np.eye(2),
                                   store_file=Path("data/2d_gaussian_1000"))


def parse_args():
    """Arg parser"""
    parser = argparse.ArgumentParser(description="Ensemble")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=10,
                        help="Number of epochs")
    parser.add_argument("--retrain",
                        action="store_true",
                        help="Retrain ensemble from scratch")
    parser.add_argument("--model_dir",
                        type=Path,
                        default="../models",
                        help="Model directory")

    return parser.parse_args()


if __name__ == "__main__":
    main()
