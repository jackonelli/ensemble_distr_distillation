""""Main entry point"""
import argparse
from pathlib import Path


def main():
    """Main"""
    args = parse_args()
    print(args)


def parse_args():
    """Arg parser"""
    parser = argparse.ArgumentParser(description="Ensemble")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=10,
                        help="Number of epochs")
    parser.add_argument("--retrain",
                        action="store_const",
                        default=True,
                        help="Retrain ensemble from scratch")
    parser.add_argument("--model_dir",
                        type=Path,
                        default="../models",
                        help="Model directory")

    return parser.parse_args()


if __name__ == "__main__":
    main()
