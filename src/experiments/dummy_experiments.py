import torch
import src.utils as utils
import src.metrics as metrics
from src.dataloaders import cifar10_ensemble_pred
from src.ensemble import tensorflow_ensemble
from src.distilled import dummy_logits_probability_distribution
from pathlib import Path
from datetime import datetime
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
from src.experiments.shifted_cmap import shifted_color_map

LOGGER = logging.getLogger(__name__)


def loss_test():
    # OBSERVATION: MIN LOSS SEEMS TO LAY AROUND TRUE ENSEMBLE LOGITS MEAN AND VARIANCE IN ALL DIMENSIONS

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = cifar10_ensemble_pred.Cifar10Data()

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    distilled_model = dummy_logits_probability_distribution.DummyLogitsProbabilityDistribution(ensemble,
                                                                                               learning_rate=args.lr,
                                                                                               scale_teacher_logits=True)
    min_mean = [-5, -2, 5, 6, 4, 5, 9, 4, -1]
    max_mean = [-2, 1, 10, 17, 9, 14, 27, 9, 3]
    min_var = [3, 2, 8, 9, 7, 6, 13, 7, 2]
    max_var = [7, 9, 15, 35, 12, 25, 80, 12, 9]

    orig_cmap = matplotlib.cm.viridis
    shifted_cmap = shifted_color_map(orig_cmap, midpoint=0.001, name='shifted')

    fig, ax = plt.subplots(2, 5)

    inputs, targets = next(iter(train_loader))
    teacher_logits = distilled_model._generate_teacher_predictions(inputs)

    par_arr = np.load("model_data/par_adam.npy", allow_pickle=True)
    mean_par = par_arr[:, :9]
    var_par = np.log(1 + np.exp(par_arr[:, 9:])) + 0.001

    for k, axis in enumerate(ax.reshape(-1)[:-1]):

        mean = np.linspace(min_mean[k], max_mean[k], 100, dtype=np.float)
        var = np.linspace(min_var[k], max_var[k], 100, dtype=np.float)

        logits = torch.unsqueeze(teacher_logits, dim=-1)[:, :, k]
        loss = np.zeros((mean.shape[0], var.shape[0]))
        for i in range(mean.shape[0]):
            for j in range(var.shape[0]):
                loss[i, j] = distilled_model.calculate_loss((torch.tensor([[mean[i]]]), torch.tensor([[var[j]]])),
                                                            logits)

        im = axis.imshow(loss, cmap=shifted_cmap, extent=[var.min(), var.max(), mean.min(), mean.max()], origin='lower')
        fig.colorbar(im, ax=axis)
        axis.plot(var_par[:, k], mean_par[:, k], 'o')

    plt.show()


def grad_test():

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = cifar10_ensemble_pred.Cifar10Data()

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    distilled_model = dummy_logits_probability_distribution.DummyLogitsProbabilityDistribution(ensemble,
                                                                                               learning_rate=args.lr,
                                                                                               scale_teacher_logits=True)

    inputs, targets = next(iter(train_loader))
    teacher_logits = distilled_model._generate_teacher_predictions(inputs)

    par_arr = np.load("model_data/par_adam.npy", allow_pickle=True)[:1000, :]

    # Adding initial par values and we don't actually care about the final parameter values
    param_dim = par_arr.shape[-1]
    par_arr = np.concatenate((np.zeros((1, param_dim)), par_arr[:-1, :]))
    mean = par_arr[:, :int(param_dim/2)]
    untr_var = par_arr[:, int(param_dim/2):]
    var = np.log(1 + np.exp(untr_var)) + 0.001

    grad_arr = np.load("model_data/grad_adam.npy", allow_pickle=True)[:1000, :]
    train_mean_grad = grad_arr[:, :int(param_dim/2)]
    train_var_grad = grad_arr[:, int(param_dim/2):]

    mean_grad = np.zeros(train_mean_grad.shape)
    untr_var_grad = np.zeros(train_var_grad.shape)
    for i in range(par_arr.shape[0]):
        mean_grad[i, :] = np.stack([calc_mean_grad(mean[i, j], var[i, j], teacher_logits[:, :, j]).mean(axis=[0, 1])
                                   for j in range(untr_var.shape[-1])])
        untr_var_grad[i, :] = np.stack([calc_z_grad(mean[i, j], var[i, j], untr_var[i, j], teacher_logits[:, :, j]).mean(axis=[0, 1])
                                        for j in range(untr_var.shape[-1])])

    fig1, ax1 = plt.subplots(5, 2)
    fig2, ax2 = plt.subplots(5, 2)

    for i, (axis1, axis2) in enumerate(zip(ax1.reshape(-1)[:-1], ax2.reshape(-1)[:-1])):
        axis1.plot(np.arange(0, par_arr.shape[0]), train_mean_grad[:, i], label="Mean grad from training")
        axis1.plot(np.arange(0, par_arr.shape[0]), mean_grad[:, i], label="Manual mean grad")
        axis1.plot(np.arange(0, par_arr.shape[0]), train_mean_grad[:, i] / mean_grad[:, i], label="Diff")
        axis1.legend()

        axis2.plot(np.arange(0, par_arr.shape[0]), train_var_grad[:, i], label="Var grad from training")
        axis2.plot(np.arange(0, par_arr.shape[0]), untr_var_grad[:, i], label="Manual var grad")
        axis2.plot(np.arange(0, par_arr.shape[0]), train_var_grad[:, i] / untr_var_grad[:, i], label="Diff")
        axis2.legend()

    fig1.suptitle("Mean")
    fig2.suptitle("Var")
    plt.show()


def calc_mean_grad(mu, sigma, target):
    return - (target - mu) / sigma


def calc_var_grad(mu, sigma, target):
    return 0.5 * ((1 / sigma) - ((target - mu) ** 2) / (sigma ** 2))


def calc_z_grad(mu, sigma, z, target):
    var_grad = calc_var_grad(mu, sigma, target)
    var_z_grad = np.exp(z) / (1 + np.exp(z))
    return var_grad * var_z_grad


def dummy_test():
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = cifar10_ensemble_pred.Cifar10Data()
    valid_set = cifar10_ensemble_pred.Cifar10Data(validation=True)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    ensemble = tensorflow_ensemble.TensorflowEnsemble(output_size=10)
    # models_dir = 'models/cifar-0508-ensemble_50/r'
    # ensemble.load_ensemble(models_dir, num_members=50)

    distilled_model = dummy_logits_probability_distribution.DummyLogitsProbabilityDistribution(ensemble,
                                                                                               learning_rate=args.lr*0.1,
                                                                                               scale_teacher_logits=True)

    loss_metric = metrics.Metric(name="Mean loss", function=distilled_model.calculate_loss)
    distilled_model.add_metric(loss_metric)

    distilled_dir = Path("models/distilled_model_cifar10_dummy")
    distilled_model.train(train_loader, num_epochs=300, validation_loader=valid_loader)
    torch.save(distilled_model, distilled_dir)


def check_dummy_test():

    train_set = cifar10_ensemble_pred.Cifar10Data()

    input_data = (torch.tensor(train_set.set.data[0]), torch.tensor(train_set.set.data[1]),
                  torch.tensor(train_set.set.data[2]))

    distilled_filepath = Path("models/distilled_model_cifar10_dummy")
    distilled_model = torch.load(distilled_filepath)

    # Look at logits
    dummy_input = torch.ones(1, 1)
    distilled_mean, distilled_var = distilled_model.forward(dummy_input)

    ensemble_logits = distilled_model._generate_teacher_predictions(input_data)
    ensemble_logits = ensemble_logits.reshape(ensemble_logits.size(0)*ensemble_logits.size(1), ensemble_logits.size(-1))   # Blir detta rätt?
    ensemble_mean, ensemble_var = np.mean(ensemble_logits.data.numpy(), axis=0), \
                                  np.var(ensemble_logits.data.numpy(), axis=0)

    #LOGGER.info
    print("Distilled model logits mean is {} and variance is {}".format(distilled_mean.data.numpy(),
                                                                        distilled_var.data.numpy()))
    print("Ensemble logits mean is {} and variance is {}".format(ensemble_mean, ensemble_var))


if __name__ == "__main__":
    #dummy_test()
    #check_dummy_test()
    loss_test()
    #grad_test()