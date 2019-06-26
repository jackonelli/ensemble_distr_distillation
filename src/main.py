""""Main entry point"""
from pathlib import Path
import numpy as np
from dataloaders import gaussian
import utils


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


if __name__ == "__main__":
    main()
