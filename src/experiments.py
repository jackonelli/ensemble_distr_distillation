import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import utils
import distilled_network
import ensemble
import gaussian
import models
from pathlib import Path

def distill_model_comparison(distill_output, ensemble_output, metric):
    """Comparison interface
    distill_output and tensor output must match in some sense
    """


def accuracy_comparison(model, ensemble, inputs, labels):
    ensemble_output = ensemble.predict(inputs)
    ensemble_prediction = torch.argmax(ensemble_output, dim=-1)
    ensemble_accuracy = (1 / inputs.shape[0]) * (ensemble_prediction == labels.type(torch.LongTensor)).sum().data.numpy()

    model_output = model.forward(inputs)
    model_prediction = torch.argmax(model_output, dim=-1)
    model_accuracy = (1 / inputs.shape[0]) * (model_prediction == labels.type(torch.LongTensor)).sum().data.numpy()

    return ensemble_accuracy, model_accuracy


def effect_of_ensemble_size():
    pass


def effect_of_model_capacity():
    pass


def entropy(p):
    return -p * torch.log(p)


def entropy_comparison(model, ensemble, inputs):
    # Comparing predictions vs entropy of ensemble and distilled model
    ensemble_output = ensemble.predict(inputs)
    ensemble_entropy = (1 / inputs.shape[0]) * torch.sum(entropy(ensemble_output),
                                                         dim=-1)

    model_output = model.forward(inputs)
    model_entropy = (1 / inputs.shape[0]) * torch.sum(
        entropy(model_output), dim=-1)  # Logiskt att kolla på detta värde?

    return ensemble_entropy.data.numpy(), model_entropy.data.numpy()


def entropy_comparison_plot(model, ensemble, inputs):
    ensemble_entropy, model_entropy = entropy_comparison(model, ensemble, inputs)

    num_bins = 100
    plt.hist(ensemble_entropy, bins=num_bins, density=True)
    plt.hist(model_entropy, bins=num_bins, density=True)
    plt.xlabel('Entropy')
    plt.legend(['Ensemble model', 'Distilled model'])
    plt.show()


def nll_comparison(model, ensemble, inputs, labels, number_of_classes):
    one_hot_labels = utils.to_one_hot(labels, number_of_classes)

    ensemble_output = ensemble.predict(inputs)
    ensemble_nll = torch.sum(-one_hot_labels * torch.log(ensemble_output))

    model_output = model.forward(inputs)
    model_nll = torch.sum(-one_hot_labels * torch.log(model_output))

    return ensemble_nll.data.numpy(), model_nll.data.numpy()


# Sen lite andra allmänna osäkerhetstest?


def noise_effect_on_entropy(model, ensemble, data):
    # Oklart om det här såhär de gör, men
    epsilon = np.linspace(0.0001, 1, 10)

    ensemble_entropy = np.zeros([
        len(epsilon),
    ])
    model_entropy = np.zeros([
        len(epsilon),
    ])
    for i, e in enumerate(epsilon):
        data_perturbed = data.copy()
        data_perturbed.x = data_perturbed.x + np.random.normal(
            loc=0, scale=epsilon, size=data.shape)
        ensemble_entropy[i], model_entropy[i] = entropy_comparison(
            model, ensemble, data_perturbed)

    plt.plot(epsilon, ensemble_entropy)
    plt.plot(epsilon, model_entropy)
    plt.xlabel('Epsilon')
    plt.ylabel('Entropy')
    plt.legend(['Ensemble model', 'Distilled model'])
    plt.show()


def ood_test(ood_data):
    # What happens with accuracy, entropy etc.?
    pass

def test():
    args = utils.parse_args()
    device = utils.torch_settings(args.seed)
    data = gaussian.SyntheticGaussianData(
        mean_0=[0, 0],
        mean_1=[-3, -3],
        cov_0=np.eye(2),
        cov_1=np.eye(2),
        store_file=Path("data/2d_gaussian_1000"))
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=0)

    prob_ensemble = ensemble.Ensemble()
    num_ensemble_members = 10
    for i in range(num_ensemble_members):
        print("Training ensemble member number {}".format(i + 1))
        model = models.NeuralNet(2, 3, 3, 2, lr=args.lr)
        model.train(train_loader, args.num_epochs)
        prob_ensemble.add_member(model)

    distilled_model = distilled_network.PlainProbabilityDistribution(
        2, 3, 3, 2, prob_ensemble, lr=0.001)

    print("Training distilled network")
    distilled_model.train(train_loader, args.num_epochs * 2, t=1.5)

    test_data = gaussian.SyntheticGaussianData(
        mean_0=[0, 0],
        mean_1=[-3, -3],
        cov_0=np.eye(2),
        cov_1=np.eye(2),
        n_samples=500,
        store_file=Path("data/2d_gaussian_test"))

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=500,
                                              shuffle=False,
                                              num_workers=0)
    data, = test_loader
    inputs = data[0]
    labels = data[1]

    ensemble_accuracy, model_accuracy = accuracy_comparison(distilled_model, prob_ensemble, inputs, labels)
    print(ensemble_accuracy)
    print(model_accuracy)

    # ensemble_nll, model_nll = experiments.nll_comparison(distilled_model, prob_ensemble, inputs, labels, 2)
    # print(ensemble_nll)
    # print(model_nll)

    entropy_comparison_plot(distilled_model, prob_ensemble, inputs)

