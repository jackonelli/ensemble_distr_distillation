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
import scipy.interpolate
from src.experiments.shifted_cmap import shifted_color_map
import matplotlib
from src.distilled import logits_matching
from src.distilled import dummy_logits_probability_distribution

LOGGER = logging.getLogger(__name__)


def get_accuracy(distilled_model):
    prob_ensemble = distilled_model.teacher

    # Check accuracy of student and teacher on test data
    test_data = gaussian.SyntheticGaussianData(mean_0=[0, 0], mean_1=[-2, -2], cov_0=np.eye(2),
                                               cov_1=np.eye(2), n_samples=1000,
                                               store_file=Path("data/2d_gaussian_1000_test"))

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1000,
                                              shuffle=True,
                                              num_workers=0)

    # Will also just check the loss on the test data
    distilled_model.calc_metrics(test_loader)

    test_inputs, test_labels = next(iter(test_loader))
    test_labels = test_labels.data.numpy()

    teacher_test_predictions = prob_ensemble.predict(test_inputs)
    teacher_predictions = torch.argmax(torch.mean(teacher_test_predictions, axis=1), axis=-1).data.numpy()
    teacher_acc = np.mean(teacher_predictions == test_labels)
    LOGGER.info("Ensemble accuracy on test data {}".format(teacher_acc))

    student_test_predictions = distilled_model.predict(test_inputs)
    student_predictions = torch.argmax(torch.stack((student_test_predictions, 1-student_test_predictions), dim=1), axis=1).data.numpy()
    student_acc = np.mean(np.transpose(student_predictions) == test_labels)
    LOGGER.info("Distilled model accuracy on test data {}".format(student_acc))


def uncertainty_plots(distilled_model):

    x_min = -4
    x_max = 3
    num_points = 1000
    x_0, x_1 = np.linspace(x_min, x_max, num_points), np.linspace(x_min, x_max, num_points)
    x_0, x_1 = np.meshgrid(x_0, x_1)
    inputs = torch.tensor(np.column_stack((x_0.reshape(num_points**2, 1), x_1.reshape(num_points**2, 1))),
                          dtype=torch.float32)

#    label_1_ind = targets == 0
#    label_2_ind = targets == 1
#    ax[0].scatter(inputs[label_1_ind, 0], inputs[label_1_ind, 1], label="Class 1", marker='.')
#    ax[0].scatter(inputs[label_2_ind, 0], inputs[label_2_ind, 1], label="Class 2", marker='.')
#    ax[0].set_title('Ground truth')

    dist_mean, dist_var = distilled_model.forward(inputs)
    dist_mean, dist_var = dist_mean.detach().numpy(), dist_var.detach().numpy()

    fig, ax = plt.subplots(2, 2)
    dist_predictions = np.argmax(np.column_stack((dist_mean, 1-dist_mean)), axis=-1)

    prob_ensemble = distilled_model.teacher
    ensemble_predicted_distribution = prob_ensemble.predict(inputs)
    ensemble_predictions = torch.argmax(torch.mean(ensemble_predicted_distribution, axis=1), axis=-1).data.numpy()
    ensemble_logits = distilled_model._generate_teacher_predictions(inputs)
    ens_var = np.var(ensemble_logits.detach().numpy(), axis=1)

    #predicted_distribution_samples = torch.mean(distilled_model.predict(inputs), axis=1)
    #distilled_model_predictions = torch.argmax(torch.stack(
    #    (predicted_distribution_samples, 1 - predicted_distribution_samples), axis=-1), axis=-1).data.numpy()
    #acc = np.mean(distilled_model_predictions == targets[:, np.newaxis])
    #LOGGER.info("Distilled model accuracy on test data {}".format(acc))

    pred_1_ind = ensemble_predictions == 0
    pred_2_ind = ensemble_predictions == 1
    ax[0, 0].imshow(ensemble_predictions.reshape(num_points, num_points),
                 extent=[inputs[:, 0].min(), inputs[:, 0].max(), inputs[:, 1].min(), inputs[:, 1].max()])
    #ax[0].scatter(inputs[pred_1_ind, 0], inputs[pred_1_ind, 1], label="Class 1", marker='.')
    #ax[0].scatter(inputs[pred_2_ind, 0], inputs[pred_2_ind, 1], label="Class 2", marker='.')
    ax[0, 0].set_title('Ensemble prediction')
    #ax[0].legend()

    pred_1_ind = dist_predictions == 0
    pred_2_ind = dist_predictions == 1
    ax[0, 1].imshow(dist_predictions.reshape(num_points, num_points),
                 extent=[inputs[:, 0].min(), inputs[:, 0].max(), inputs[:, 1].min(), inputs[:, 1].max()])
    #ax[1].scatter(inputs[pred_1_ind, 0], inputs[pred_1_ind, 1], label="Class 1", marker='.')
    #ax[1].scatter(inputs[pred_2_ind, 0], inputs[pred_2_ind, 1], label="Class 2", marker='.')
    ax[0, 1].set_title('Distilled model distribution mean')
    #ax[1].legend()

    #tot_unc, ep_unc, al_unc = metrics.uncertainty_separation_entropy(ensemble_predicted_distribution, None)

    #ens_var_rbf = scipy.interpolate.Rbf(inputs[:, 0], inputs[:, 1], ens_var, function='linear')(xi, yi)
    #dist_var_rbf = scipy.interpolate.Rbf(inputs[:, 0], inputs[:, 1], var, function='linear')(xi, yi)

    var_min = np.minimum(ens_var.min(), dist_var.min())
    var_max = np.maximum(ens_var.max(), dist_var.max())

    orig_cmap = matplotlib.cm.viridis
    shifted_cmap = shifted_color_map(orig_cmap, midpoint=0.05, name='shifted')
    ens_var = ax[1, 0].imshow(ens_var.reshape(num_points, num_points), cmap=shifted_cmap, vmin=var_min, vmax=var_max,
                           extent=[inputs[:, 0].min(), inputs[:, 0].max(), inputs[:, 1].min(), inputs[:, 1].max()])
    ax[1, 0].set_title('Ensemble model logits variance')
    fig.colorbar(ens_var, ax=ax[1, 0])
    dist_var = ax[1, 1].imshow(dist_var.reshape(num_points, num_points), cmap=shifted_cmap, vmin=var_min, vmax=var_max,
                            extent=[inputs[:, 0].min(), inputs[:, 0].max(), inputs[:, 1].min(), inputs[:, 1].max()])
    ax[1, 1].set_title('Distilled model logits variance')
    fig.colorbar(dist_var, ax=ax[1, 1])

    plt.show()


def distribution_test(prob_ensemble, distilled_model):

    # OBS! Blir bara endimesnionellt nu
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

    # for _ in range(args.num_ensemble_members * 10):
    #     model = simple_classifier.SimpleClassifier(input_size,
    #                                                hidden_size,
    #                                                hidden_size,
    #                                                ensemble_output_size,
    #                                                device=device,
    #                                                learning_rate=args.lr)
    #     prob_ensemble.add_member(model)
    #
    # acc_metric = metrics.Metric(name="Acc", function=metrics.accuracy)
    # loss_metric = metrics.Metric(name="Mean val loss", function=model.calculate_loss)
    # prob_ensemble.add_metrics([acc_metric, loss_metric])
    # prob_ensemble.train(train_loader, args.num_epochs, validation_loader=validation_loader)

    ensemble_filepath = Path("models/simple_class_logits_ensemble_overlap_50")
    #
    #prob_ensemble.save_ensemble(ensemble_filepath)
    prob_ensemble.load_ensemble(ensemble_filepath)

    full_data = gaussian.SyntheticGaussianData(
        mean_0=[0, 0],
        mean_1=[-2, -2],
        cov_0=np.eye(2),
        cov_1=np.eye(2),
        sample=False,
        store_file=Path("data/2d_gaussian_full"))

    full_train_loader = torch.utils.data.DataLoader(full_data,
                                                    batch_size=10,
                                                    shuffle=True,
                                                    num_workers=0)

    distilled_output_size = 2
    #
    #distilled_model = logits_matching.LogitsMatching(
    distilled_model = logits_probability_distribution.LogitsProbabilityDistribution(
        input_size,
        hidden_size,
        hidden_size,
        distilled_output_size,
        teacher=prob_ensemble,
        device=device,
        learning_rate=args.lr*0.1)

    loss_metric = metrics.Metric(name="Mean val loss", function=distilled_model.calculate_loss)
    distilled_model.add_metric(loss_metric)
    distilled_filepath = Path("models/simple_class_logits_distilled_overlap_full_training")
    distilled_model = torch.load(distilled_filepath)
    #distilled_model.train(full_train_loader, args.num_epochs*10, validation_loader=validation_loader)

    #distilled_filepath = Path("models/simple_class_logits_distilled_matching_logits_one_ensemble_member")
    distilled_filepath = Path("models/simple_class_logits_distilled_overlap_full_training")
    #distilled_model = torch.load(distilled_filepath)
    distilled_model.calc_metrics(full_train_loader)


    #
    torch.save(distilled_model, distilled_filepath)
    #distilled_model = torch.load(distilled_filepath)
    get_accuracy(distilled_model)
    uncertainty_plots(distilled_model)
    # distribution_test(prob_ensemble, distilled_model)


if __name__ == "__main__":
    main()
