""""Classification experiment with synthetic data and logits matching"""
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import torch
import sys
print(sys.path)

from src.dataloaders import gaussian
import src.utils as utils
from src.distilled import logits_probability_distribution
from src.ensemble import ensemble
from src.ensemble import simple_classifier
import src.metrics as metrics
import matplotlib.pyplot as plt
import torch
from scipy.stats import multivariate_normal as scipy_mvn
from scipy.stats import norm as scipy_norm

LOGGER = logging.getLogger(__name__)


def uncertainty_plots(prob_ensemble, distilled_model):

    data = gaussian.SyntheticGaussianData(mean_0=[0, 0], mean_1=[-2, -2], cov_0=np.eye(2),
                                          cov_1=np.eye(2), n_samples=500, store_file=Path("data/one_dim_reg_500"))

    test_loader = torch.utils.data.DataLoader(data,
                                              batch_size=500,
                                              shuffle=True,
                                              num_workers=0)

    prob_ensemble.calc_metrics(test_loader)

    inputs, targets = next(iter(test_loader))
    targets = targets.data.numpy()

    fig, ax = plt.subplots(4, 2)

    label_1_ind = targets == 0
    label_2_ind = targets == 1
    ax[0, 0].scatter(inputs[label_1_ind, 0], inputs[label_1_ind, 1], label="Class 1", marker='.')
    ax[0, 0].scatter(inputs[label_2_ind, 0], inputs[label_2_ind, 1], label="Class 2", marker='.')
    ax[0, 0].set_title('Ground truth')

    mean, var = distilled_model.forward(inputs)
    mean, var = mean.detach().numpy(), var.detach().numpy()

    predictions = np.argmax(mean, axis=-1)
    pred_1_ind = predictions == 0
    pred_2_ind = predictions == 1
    ax[0, 1].scatter(inputs[pred_1_ind, 0], inputs[pred_1_ind, 1], label="Class 1", marker='.')
    ax[0, 1].scatter(inputs[pred_2_ind, 0], inputs[pred_2_ind, 1], label="Class 2", marker='.')
    ax[0, 1].set_title('Distilled model distribution mean')

    ensemble_predicted_distribution = prob_ensemble.predict(inputs)
    ensemble_predictions = torch.argmax(torch.mean(ensemble_predicted_distribution, axis=1), axis=-1).data.numpy()
    acc = np.mean(ensemble_predictions == targets)
    LOGGER.info("Ensemble accuracy on test data {}".format(acc))
    model_uncertainty(ensemble_predicted_distribution, inputs, ax[1, :], "Ensemble")

    predicted_distribution_samples = distilled_model.predict(inputs)
    distilled_model_predictions = torch.argmax(torch.mean(predicted_distribution_samples, axis=1), axis=-1).data.numpy()
    acc = np.mean(distilled_model_predictions == targets)
    LOGGER.info("Distilled model accuracy on test data {}".format(acc))
    model_uncertainty(predicted_distribution_samples, inputs.data.numpy(), ax[2, :], "Distilled model")

    ax[3, 0].scatter(inputs[:, 0], var[:, 0])
    ax[3, 0].set_title('Distilled model logits variance, dim 1')
    ax[3, 1].scatter(inputs[:, 1], var[:, 1])
    ax[3, 0].set_title('Distilled model logits variance, dim 2')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[2, 0].legend()
    ax[2, 1].legend()

    plt.show()


def distribution_test(prob_ensemble, distilled_model):
    data = gaussian.SyntheticGaussianData(mean_0=[0, 0], mean_1=[-2, -2], cov_0=np.eye(2),
                                          cov_1=np.eye(2), n_samples=6, store_file=Path("data/one_dim_reg_6"))

    test_loader = torch.utils.data.DataLoader(data,
                                              batch_size=data.n_samples,
                                              shuffle=True,
                                              num_workers=0)

    inputs, labels = next(iter(test_loader))

    ensemble_logits_samples = prob_ensemble.get_logits(inputs).detach().numpy()
    distilled_mean, distilled_var = distilled_model.forward(inputs)
    distilled_mean, distilled_var = distilled_mean.detach().numpy(), distilled_var.detach().numpy()

    fig, axes = plt.subplots(int(data.n_samples / 2), 2)

    for i, ax in enumerate(axes.reshape(-1)):

        rv = scipy_mvn(distilled_mean[i, :], np.diag(distilled_var[i, :]))
        min_x = scipy_norm.ppf(0.01, distilled_mean[i, 0], distilled_var[i, 0])
        min_y = scipy_norm.ppf(0.01, distilled_mean[i, 1], distilled_var[i, 1])
        max_x = scipy_norm.ppf(0.99, distilled_mean[i, 0], distilled_var[i, 0])
        max_y = scipy_norm.ppf(0.99, distilled_mean[i, 1], distilled_var[i, 1])

        x, y = np.mgrid[min_x:max_x:((max_x-min_x)/100), min_y:max_y:((max_y-min_y)/100)]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        ax.contourf(x, y, rv.pdf(pos))

        for j in range(len(prob_ensemble.members)):
            ax.scatter(ensemble_logits_samples[i, j, 0], ensemble_logits_samples[i, j, 1])

        ax.set_title("Input: " + str(inputs[i, :].data.numpy()))

    plt.show()


def model_uncertainty(predicted_distribution, test_data, ax, key):

    tot_unc, ep_unc, al_unc = metrics.uncertainty_separation_entropy(predicted_distribution, None)

    ax[0].scatter(test_data[:, 0], tot_unc.detach().numpy(), label="Total uncertainty")
    ax[0].scatter(test_data[:, 0], ep_unc.detach().numpy(), label="Epistemic uncertainty", marker='.')
    ax[0].scatter(test_data[:, 0], tot_unc.detach().numpy(), label="Aleatoric uncertainty", marker='.')
    ax[0].set_title(key + " uncertainty (entropy-based), dim 1")

    ax[1].scatter(test_data[:, 1], tot_unc.detach().numpy(), label="Total uncertainty")
    ax[1].scatter(test_data[:, 1], ep_unc.detach().numpy(), label="Epistemic uncertainty", marker='.')
    ax[1].scatter(test_data[:, 1], tot_unc.detach().numpy(), label="Aleatoric uncertainty", marker='.')
    ax[1].set_title(key + " uncertainty (entropy-based), dim 2")


def main():
    """Main"""
    args = utils.parse_args()
    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)
    LOGGER.info("Args: {}".format(args))
    device = utils.torch_settings(args.seed, args.gpu)
    LOGGER.info("Creating dataloader")
    data = gaussian.SyntheticGaussianData(
        mean_0=[0, 0],
        mean_1=[-2, -2],
        cov_0=np.eye(2),
        cov_1=np.eye(2),
        store_file=Path("data/2d_gaussian_1000"))

    val_data = gaussian.SyntheticGaussianData(
            mean_0=[0, 0],
            mean_1=[-2, -2],
            cov_0=np.eye(2),
            cov_1=np.eye(2),
            n_samples=500,
            store_file=Path("data/2d_gaussian_500"))

    # TODO: Automated dims
    input_size = 2
    hidden_size = 3
    ensemble_output_size = 2

    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=0)
    validation_loader = torch.utils.data.DataLoader(val_data,
                                                    batch_size=4,
                                                    shuffle=True,
                                                    num_workers=0)

    prob_ensemble = ensemble.Ensemble(ensemble_output_size)

    for _ in range(args.num_ensemble_members * 10):
        model = simple_classifier.SimpleClassifier(input_size,
                                                   hidden_size,
                                                   hidden_size,
                                                   ensemble_output_size,
                                                   device=device,
                                                   learning_rate=args.lr)
        prob_ensemble.add_member(model)

    acc_metric = metrics.Metric(name="Acc", function=metrics.accuracy)
    loss_metric = metrics.Metric(name="Mean val loss", function=model.calculate_loss)
    prob_ensemble.add_metrics([acc_metric, loss_metric])
    prob_ensemble.train(train_loader, args.num_epochs, validation_loader=validation_loader)

    ensemble_filepath = Path("models/simple_class_logits_ensemble_overlap_50")
    #
    prob_ensemble.save_ensemble(ensemble_filepath)
    #prob_ensemble.load_ensemble(ensemble_filepath)

    distilled_output_size = 2
    distilled_model = logits_probability_distribution.LogitsProbabilityDistribution(
        input_size,
        hidden_size,
        hidden_size,
        distilled_output_size,
        teacher=prob_ensemble,
        device=device,
        learning_rate=args.lr * 0.1)

    loss_metric = metrics.Metric(name="Mean val loss", function=distilled_model.calculate_loss)
    distilled_model.add_metric(loss_metric)
    distilled_model.train(train_loader, args.num_epochs * 2, validation_loader=validation_loader)

    distilled_filepath = Path("models/simple_class_logits_distilled_overlap")

    #torch.save(distilled_model, distilled_filepath)
    distilled_model = torch.load(distilled_filepath)
    uncertainty_plots(prob_ensemble, distilled_model)
    distribution_test(prob_ensemble, distilled_model)


if __name__ == "__main__":
    main()
