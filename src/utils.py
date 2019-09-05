"""Utilities module"""
import sys
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def to_one_hot(labels, number_of_classes):
    """Labels is a tensor of class indices"""
    return nn.functional.one_hot(labels, number_of_classes)


def torch_settings(seed=1, use_gpu=False):
    """Pytorch settings"""
    if torch.cuda.is_available() and use_gpu:
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
    parser.add_argument("--num_ensemble_members",
                        type=int,
                        default=5,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--retrain",
                        action="store_true",
                        help="Retrain ensemble from scratch")
    parser.add_argument("--model_dir",
                        type=Path,
                        default="./models",
                        help="Model directory")
    parser.add_argument("--log_dir",
                        type=Path,
                        default="./logs",
                        help="Logs directory")
    parser.add_argument("--log_level",
                        type=_log_level_arg,
                        default=logging.INFO,
                        help="Log level")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="Random seed, NB both cuda and cpu")
    parser.add_argument("--gpu",
                        action="store_true",
                        help="Use gpu, if available")

    return parser.parse_args()


def _log_level_arg(arg_string):
    arg_string = arg_string.upper()
    if arg_string == "DEBUG":
        log_level = logging.DEBUG
    elif arg_string == "INFO":
        log_level = logging.INFO
    elif arg_string == "WARNING":
        log_level = logging.WARNING
    elif arg_string == "ERROR":
        log_level = logging.WARNING
    elif arg_string == "CRITICAL":
        log_level = logging.WARNING
    else:
        raise ValueError("Invalid log level.")
    return log_level


LOG_FORMAT = "%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s"


def setup_logger(log_path=None,
                 logger=None,
                 log_level=logging.INFO,
                 fmt=LOG_FORMAT):
    """Setup for a logger instance.

    Args:
        log_path (str, optional): full path
        logger (logging.Logger, optional): root logger if None
        log_level (logging.LOGLEVEL, optional):
        fmt (str, optional): message format

    """
    logger = logger if logger else logging.getLogger()
    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.setLevel(log_level)
    logger.handlers = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    log_path = Path(log_path)
    if log_path:
        directory = log_path.parent
        directory.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Log at {}".format(log_path))


def hard_classification(predicted_distribution):
    """Hard classification from forwards' probability distribution
    """
    class_ind, confidence = tensor_argmax(predicted_distribution)
    return class_ind, confidence


def tensor_argmax(input_tensor):
    value, ind = torch.max(input_tensor, dim=-1)
    return ind, value
