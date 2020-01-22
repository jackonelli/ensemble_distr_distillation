"""Utilities module"""
import sys
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import math


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ScaleTransform:
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, img):
        return img / self.scaling_factor


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
    parser.add_argument("--saved_model",
                        type=_saved_model_path_arg,
                        default=None,
                        help="Path to saved model")
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
        raise argparse.ArgumentTypeError(
            "Invalid log level: {}".format(arg_string))
    return log_level


def _saved_model_path_arg(arg_string):
    model_path = Path(arg_string)
    if not model_path.exists():
        raise argparse.ArgumentTypeError(
            "Saved model does not exist: {}".format(model_path))
    return model_path


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


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):
    """ Cyclical learning rate
    the torch_optim.lr_scheduler.CycleLR does not work with Adam,
    instead I copied this one from here:
    https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
    """

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def variance_linear_asymptote(input_):
    """Variance transform
    Element-wise map of input_ input to positive real axis

    Asymptotically linear in input for large inputs
    """
    return torch.log(1 + torch.exp(input_))


def variance_exponential(input_):
    """Variance transform
    Element-wise map of input_ input to positive real axis
    """
    return torch.exp(input_)


def variance_moberg(input_, epsilon=1e-6):
    """Variance transform
    Element-wise map of input_ input to positive real axis
    As used in John Moberg's thesis
    """

    return torch.log(1 + torch.exp(input_) + epsilon) + epsilon


def gradient_analysis(model):
    """Extracts some gradient characteristics from a training model"""
    gradients = [param.grad for param in model.layers[-1].parameters()]
    return gradients
