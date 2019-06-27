"""Utilities module"""
import argparse
from pathlib import Path
import torch


def torch_settings(seed=1):
    """Pytorch settings"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    return device


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

    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="Random seed, NB both cuda and cpu")

    return parser.parse_args()
