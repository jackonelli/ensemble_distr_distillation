"""Utilities module"""
import sys
import logging
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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


def adapted_lr(c=0.7):
    # the torch_optim.lr_scheduler.CycleLR does not work with Adam so I copied this one from here:
    # https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee

    # Lambda function to calculate the LR
    lr_lambda = lambda it: (it + 1)**(-c)

    return lr_lambda


def variance_linear_asymptote(epsilon=0.0):
    """Variance transform
    Element-wise map of input_ input to positive real axis

    Asymptotically linear in input for large inputs

    Args:
        epsilon (float): Small positive offset for numerical stability
    """
    return lambda input_: torch.log(1 + torch.exp(input_)) + epsilon


def variance_exponential(epsilon=0.0):
    """Variance transform
    Element-wise map of input_ input to positive real axis

    Args:
        epsilon (float): Small positive offset for numerical stability
    """
    return lambda input_: torch.exp(input_) + epsilon


def variance_moberg(epsilon=0.0):
    """Variance transform
    Element-wise map of input_ input to positive real axis
    As used in John Moberg's thesis

    Args:
        epsilon (float): Small positive offset for numerical stability
    """

    return lambda input_: torch.log(1 + torch.exp(input_) + epsilon) + epsilon


def generate_order(arr, descending=True):
    """Generate order based on array"""
    sorted_indices = np.argsort(arr)
    if descending:
        sorted_indices = sorted_indices[::-1]
    return sorted_indices.reshape((len(arr), ))


def moving_average(arr, window):
    """Moving average"""
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def plot_error_curve(ax, y_true, y_pred, uncert_meas, label, window_size=10):
    """Plot errors sorted according to uncertainty measure"""
    error = (y_true - y_pred)**2
    error, uncert_meas = np.array(error), np.array(uncert_meas)
    sorted_inds = generate_order(uncert_meas)
    sorted_error = error[sorted_inds]
    smooth_error = moving_average(sorted_error, window_size)
    ax.plot(np.arange(len(smooth_error)), smooth_error, label=label)


def shifted_color_map(cmap,
                      start=0,
                      midpoint=0.5,
                      stop=1.0,
                      name='shiftedcmap'):
    '''
    function taken from
    https://stackoverflow.com/questions/7404116/...
        ...defining-the-midpoint-of-a-colormap-in-matplotlib
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def gaussian_mixture_moments(mus, sigma_sqs):
    """Estimate moments of a gaussian mixture model

    B - number of observations/samples
    N - number of components in mixture

    Args:
        mus torch.tensor((B, N)): Collection of mu-values
        sigma_sqs torch.tensor((B, N)): Collection of sigma_sq-values
    """

    with torch.no_grad():
        mu = torch.mean(mus, dim=1)
        sigma_sq = torch.mean(sigma_sqs + mus**2, dim=1) - mu**2

    return mu, sigma_sq


def torch_cov(arr, rowvar=False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        arr: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `arr` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """

    if arr.dim() > 2:
        raise ValueError('arr has more than 2 dimensions')
    if arr.dim() < 2:
        arr = arr.view(1, -1)
    if not rowvar and arr.size(0) != 1:
        arr = arr.t()
    # arr = arr.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (arr.size(1) - 1)
    arr -= torch.mean(arr, dim=1, keepdim=True)
    arr_transposed = arr.t()  # if complex: mt = m.t().conj()
    return fact * arr.matmul(arr_transposed).squeeze()
