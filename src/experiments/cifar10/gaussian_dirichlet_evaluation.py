# Checkin Gaussian and Dirichlet assumptions
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import scipy.stats as scipy_stats

from src import utils
from src.dataloaders import cifar10_ensemble_pred
import dirichlet

# This will help us handle warnings as errors
import warnings
warnings.filterwarnings("error")

LOGGER = logging.getLogger(__name__)


def check_dirichlet(train_samples, test_samples):
    """
    train_samples: NxK Dirichlet distributed samples
    test_samples: NxK Dirichlet distributed samples
    return: likelihood over test_samples given the model fitted to train_samples
    """

    alpha = dirichlet.mle(train_samples, method="fixed_point") # Find by numerical means

    log_likelihood = 0
    for i in range(test_samples.shape[0]):
        log_likelihood += scipy_stats.dirichlet.logpdf(test_samples[i, :], alpha=alpha)

    return log_likelihood


def check_gaussian(train_samples, test_samples):
    """
    train_samples: Nx(K-1) Gaussian distributed samples
    test_samples: NxK logistic-normal distributed samples
    return: likelihood over test_samples given the model fitted to train_samples
    """

    mean = np.mean(train_samples, axis=0)
    var = np.var(train_samples, axis=0)

    factor = (mean.shape[0] / 2) * np.log(2*np.pi) - np.sum(np.log(test_samples), axis=-1)

    log_scaled_samples = np.log(test_samples[:, :-1] / test_samples[:, -1][:, np.newaxis])
    log_likelihood = np.sum(factor + scipy_stats.multivariate_normal.logpdf(log_scaled_samples, mean=mean,
                                                                            cov=np.diag(var)))

    return log_likelihood


def main():
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    data_ind = np.load("data/training_data_indices.npy")
    num_data_points = 1000
    data_ind = data_ind[:num_data_points]

    data_set = cifar10_ensemble_pred.Cifar10Data(ind=data_ind)

    ensemble_logits = data_set.set.data[2]
    scaled_ensemble_logits = ensemble_logits - ensemble_logits[:, :, -1][:, :, np.newaxis]
    ensemble_logits = scaled_ensemble_logits[:, :, :-1]
    ensemble_predictions = data_set.set.data[1]

    num_distributions = ensemble_logits.shape[0]
    num_samples = ensemble_logits.shape[1]

    likelihood_ratio = 0
    for i in range(num_distributions):
        train_inds = np.random.choice(num_samples, size=int(num_samples / 2), replace=False)
        test_inds = np.stack([i for i in np.arange(num_samples) if i not in train_inds])

        # We do this in the same place since likelihoods are only comparable if the same data is used
        # Will this count as the same data? I guess if I use logit-Normal distribution?

        # Gaussian
        gaussian_log_likelihood = check_gaussian(ensemble_logits[i, train_inds, :], ensemble_predictions[i, test_inds, :])

        # Dirichlet
        try:
            dirichlet_log_likelihood = check_dirichlet(ensemble_predictions[i, train_inds, :],
                                                       ensemble_predictions[i, test_inds, :])
            # Find likelihood ratio
            likelihood_ratio += np.exp(dirichlet_log_likelihood - gaussian_log_likelihood)
            print(dirichlet_log_likelihood - gaussian_log_likelihood)

        except (dirichlet.dirichlet.NotConvergingError, RuntimeWarning):
            LOGGER.info("Exception occured at iteration {}".format(i))
            num_distributions -= 1

    print(num_distributions)
    likelihood_ratio /= num_distributions
    LOGGER.info("Mean likelihood ratio is {}".format(likelihood_ratio))


if __name__ == '__main__':
    main()
