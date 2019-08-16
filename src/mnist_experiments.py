import math
import numpy as np
import torch
import torchvision
import logging
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime
import distilled_network
import ensemble
import metrics
import models
import utils

LOGGER = logging.getLogger(__name__)


def create_distilled_model(
        train_loader,
        test_loader,
        args,
        ensemble,
        filepath,
        class_type=distilled_network.PlainProbabilityDistribution):
    """Create a distilled network trained with ensemble output"""

    input_size = 784
    hidden_size_1 = 54
    hidden_size_2 = 32
    output_size = 10

    distilled_model = class_type(input_size, hidden_size_1, hidden_size_2, output_size, ensemble,
                                 learning_rate=args.lr*10)

    distilled_model.train(train_loader, args.num_epochs, t=1)
    LOGGER.info("Distilled model accuracy on test data: {}".format(get_accuracy_iter(distilled_model, test_loader)))

    torch.save(distilled_model, filepath)

    return distilled_model


def create_ensemble(train_loader, test_loader, args, num_ensemble_members, filepath):
    """Create an ensemble model"""

    input_size = 784
    hidden_size_1 = 54
    hidden_size_2 = 32
    output_size = 10

    prob_ensemble = ensemble.Ensemble()

    for i in range(num_ensemble_members):
        LOGGER.info("Training ensemble member number {}".format(i + 1))
        model = models.NeuralNet(input_size,
                                 hidden_size_1,
                                 hidden_size_2,
                                 output_size,
                                 learning_rate=args.lr)
        model.train(train_loader, args.num_epochs)
        LOGGER.info("Accuracy on test data: {}".format(
            get_accuracy_iter(model, test_loader)))
        prob_ensemble.add_member(model)

    LOGGER.info("Ensemble accuracy on test data: {}".format(
        get_accuracy_iter(prob_ensemble, test_loader)))

    prob_ensemble.save_ensemble(filepath)

    return prob_ensemble


def entropy_comparison_rotation(prob_ensemble, distilled_model, test_sample):
    """Compare the entropy of ensemble and model on the output of a rotated data point"""

    test_img = test_sample[0][0].view(28, 28)
    test_label = test_sample[1][0]

    num_points = 10
    max_val = 90
    angles = torch.arange(-max_val, max_val, max_val * 2 / num_points)

    rotated_data_set = generate_rotated_data_set(test_img, angles)
    plot_data_set(rotated_data_set)

    rotated_data_set = torch.stack(
        [data_point.view(28 * 28) for data_point in rotated_data_set])
    ensemble_member = prob_ensemble.members[0]
    ensemble_rotation_entropy, ensemble_rotation_prediction = get_entropy(
        prob_ensemble, rotated_data_set)
    ensemble_member_rotation_entropy, ensemble_member_rotation_prediction = \
        get_entropy(ensemble_member, rotated_data_set)
    distilled_model_rotation_entropy, distilled_model_rotation_prediction = \
        get_entropy(distilled_model, rotated_data_set)

    LOGGER.info("True label is: {}".format(test_label))
    LOGGER.info("Ensemble prediction: {}".format(
        ensemble_rotation_prediction))
    LOGGER.info("Ensemble member prediction: {}".format(
        ensemble_member_rotation_prediction))
    LOGGER.info("Distilled model predictions: {}".format(
        distilled_model_rotation_prediction))

    angles = angles.data.numpy()
    plt.plot(angles, ensemble_rotation_entropy.data.numpy(), 'o--')
    plt.plot(angles, ensemble_member_rotation_entropy.data.numpy(), 'o-')
    plt.plot(angles, distilled_model_rotation_entropy.data.numpy(), 'o-')
    plt.xlabel('Rotation angle')
    plt.ylabel('Entropy')
    plt.legend(["Ensemble", "Ensemble member", "Distilled model"])
    plt.show()


def dirichlet_test(train_loader, test_loader, args, ensemble):
    """Train a distilled network that uses a Dirichlet distribution for prediction"""

    filepath = Path("models/distilled_model_dirichlet_best_yet_test_t1")
    #distilled_model_dirichlet = create_distilled_model(
     #   train_loader, test_loader, args, ensemble, filepath,
      #  distilled_network.DirichletProbabilityDistribution)

    distilled_model_dirichlet = torch.load(filepath)

    distilled_model_dirichlet.train(train_loader, args.num_epochs*2, t=1)
    torch.save(distilled_model_dirichlet, Path("models/distilled_model_dirichlet_best_yet_test_t1_more_training"))

    LOGGER.info("Distilled model accuracy on train data: {}".format(get_accuracy_iter(distilled_model_dirichlet,
                                                                                      train_loader)))
    LOGGER.info("Accuracy on test data: {}".format(get_accuracy_iter(distilled_model_dirichlet, test_loader)))


def generate_rotated_data_set(img, angles):
    """Generate a set of rotated images from a single image
    Args:
        img (tensor(dim=2)): image to be rotated
        angles (tensor/ndarray): set of angles for which the image should be rotated
    """
    img = torchvision.transforms.ToPILImage()(img)
    data_set = [
        torch.squeeze(torchvision.transforms.ToTensor()(
            torchvision.transforms.functional.rotate(img, angle=angle)))
        for angle in angles
    ]

    return data_set


def get_accuracy(model, inputs, labels):
    """Calculate error of model on data set"""

    predicted_distribution = model.predict(inputs)
    accuracy = metrics.accuracy(labels, predicted_distribution)

    return accuracy


def get_accuracy_iter(model, data_loader):
    """Calculate accuracy of model on data in dataloader"""

    accuracy = 0
    num_batches = 0
    for batch in data_loader:
        inputs, labels = batch
        accuracy += get_accuracy(model, inputs, labels)
        num_batches += 1

    return accuracy / num_batches


def get_error_iter(model, data_loader):
    """Calculate error of model on data in dataloader"""

    error = 0
    num_batches = 0
    for batch in data_loader:
        inputs, labels = batch
        error += (1 - get_accuracy(model, inputs, labels))
        num_batches += 1

    return error / num_batches


def get_entropy(model, data_set):
    """Calculate entropy of model output over a data set"""

    output = model.predict(data_set)
    entropy = metrics.entropy(output)

    prediction = torch.max(output, dim=-1)

    return entropy, prediction


def get_entropy_iter(model, test_loader):
    """Calculate entropy of model output over a dataloader"""

    entropy = []
    for i, batch in enumerate(test_loader):
        inputs, labels = batch
        if i == 0:
            entropy, _ = get_entropy(model, inputs)

        else:
            batch_entropy, _ = get_entropy(model, inputs)
            entropy = torch.cat((entropy, batch_entropy), dim=0)

    return entropy


def load_mnist_data(batch_size_train=32, batch_size_test=32):
    """Loading of MNIST data set to dataloaders"""

    mnist_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(
        ),  # torchvision.transforms.Normalize((0.1307,), (0.3081,)
        utils.ReshapeTransform((-1, ))
    ])

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        '/files/', train=True, download=True, transform=mnist_transform),
                                               batch_size=batch_size_train,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        '/files/', train=False, download=True, transform=mnist_transform),
                                              batch_size=batch_size_test,
                                              shuffle=True)

    return train_loader, test_loader


def noise_effect_on_entropy(model, ensemble, test_loader):
    """Effect on entropy of ensemble and model with increasing noise added to the input"""

    ensemble_member = ensemble.members[0]

    epsilon = torch.linspace(0.0001, 1, 10)
    ensemble_entropy = torch.zeros([
        len(epsilon),
    ])
    ensemble_member_entropy = torch.zeros([
        len(epsilon),
    ])
    model_entropy = torch.zeros([
        len(epsilon),
    ])

    for i, e in enumerate(epsilon):

        for batch in test_loader:
            inputs, labels = batch

            distr = torch.distributions.Normal(0, e)
            input_perturbed = inputs + distr.sample(
                (inputs.shape[0], inputs.shape[1]))

            entropy, _ = get_entropy(ensemble, input_perturbed)
            ensemble_entropy[i] += torch.sum(entropy)
            entropy, _ = get_entropy(ensemble_member, input_perturbed)
            ensemble_member_entropy[i] += torch.sum(entropy)
            entropy, _ = get_entropy(model, input_perturbed)
            model_entropy[i] += torch.sum(entropy)

    epsilon = epsilon.data.numpy()
    plt.plot(epsilon, ensemble_entropy.data.numpy())
    plt.plot(epsilon, ensemble_member_entropy.data.numpy())
    plt.plot(epsilon, model_entropy.data.numpy())
    plt.xlabel('\u03B5')
    plt.ylabel('Entropy')
    plt.legend(['Ensemble model', 'Ensemble member', 'Distilled model'])
    plt.show()


def effect_of_ensemble_size(full_ensemble, train_loader, test_loader, args):
    """Effect of ensemble size on error and nll of distilled model"""

    ensemble_size = len(full_ensemble.members)
    ensemble_member = full_ensemble.members[0]

    ensemble_error = torch.zeros([ensemble_size, ])
    ensemble_member_error = torch.zeros([ensemble_size, ])
    distilled_model_error = torch.zeros([ensemble_size, ])

    ensemble_nll = torch.zeros([ensemble_size, ])
    ensemble_member_nll = torch.zeros([ensemble_size, ])
    distilled_model_nll = torch.zeros([ensemble_size, ])

    concat_ensemble = ensemble.Ensemble()
    for i in range(ensemble_size):
        LOGGER.info("Number of ensemble members: {}".format(i + 1))
        concat_ensemble.add_member(full_ensemble.members[i])

        filepath = Path(
            "models/distilled_models_concat/distilled_model_ensemble_size_{}".
            format(i + 1))
        distilled_model = torch.load(
            filepath
        )  # create_distilled_model(train_loader, test_loader, args, concat_ensemble, filepath)

        ensemble_error[i] = get_error_iter(concat_ensemble, test_loader)
        ensemble_member_error[i] = get_error_iter(ensemble_member, test_loader)
        distilled_model_error[i] = get_error_iter(distilled_model, test_loader)
        ensemble_nll[i] = torch.sum(
            get_entropy_iter(concat_ensemble, test_loader))
        ensemble_member_nll[i] = torch.sum(
            get_entropy_iter(ensemble_member, test_loader))
        distilled_model_nll[i] = torch.sum(
            get_entropy_iter(distilled_model, test_loader))

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.arange(ensemble_size), ensemble_error.data.numpy())
    axes[0].plot(np.arange(ensemble_size), ensemble_member_error.data.numpy())
    axes[0].plot(np.arange(ensemble_size), distilled_model_error.data.numpy())
    axes[0].set_xlabel('Ensemble size')
    axes[0].set_ylabel('Error')
    axes[0].legend(['Ensemble', 'Ensemble member', 'Distilled model'])

    axes[1].plot(np.arange(ensemble_size), ensemble_nll.data.numpy())
    axes[1].plot(np.arange(ensemble_size), ensemble_member_nll.data.numpy())
    axes[1].plot(np.arange(ensemble_size), distilled_model_nll.data.numpy())
    axes[1].set_xlabel('Ensemble size')
    axes[1].set_ylabel('NLL')
    axes[1].legend(['Ensemble', 'Ensemble member', 'Distilled model'])
    plt.show()


def entropy_histogram(ensemble, model, test_loader):
    """Comparison of entropy histograms of ensemble and model"""

    ensemble_member = ensemble.members[0]

    ensemble_entropy = get_entropy_iter(ensemble, test_loader).data.numpy()
    ensemble_member_entropy = get_entropy_iter(ensemble_member, test_loader).data.numpy()
    distilled_model_entropy = get_entropy_iter(model, test_loader).data.numpy()

    num_bins = 100
    plt.hist(ensemble_entropy, bins=num_bins, alpha=0.5, density=True)
    plt.hist(ensemble_member_entropy, bins=num_bins, alpha=0.5, density=True)
    plt.hist(distilled_model_entropy, bins=num_bins, alpha=0.5, density=True)
    plt.xlabel('Entropy')
    plt.legend(['Ensemble model', 'Ensemble member', 'Distilled model'])

    plt.show()

    # Detta verkar ge ganska lika resultat? Men för det enkla 2d-fallet observerade jag skillnad för en mindre ensemble-size


def plot_data_set(data_set):
    """Plot of image data set
    Args:
        data_set (list(len=10)): list of ten images/2D ndarrays
    """

    fig, axes = plt.subplots(2, 5)
    axes[0, 0].imshow(data_set[0])
    axes[0, 1].imshow(data_set[1])
    axes[0, 2].imshow(data_set[2])
    axes[0, 3].imshow(data_set[3])
    axes[0, 4].imshow(data_set[4])
    axes[1, 0].imshow(data_set[5])
    axes[1, 1].imshow(data_set[6])
    axes[1, 2].imshow(data_set[7])
    axes[1, 3].imshow(data_set[8])
    axes[1, 4].imshow(data_set[9])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def main():
    """Main"""

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_loader, test_loader = load_mnist_data()

    #num_ensemble_members = 10

    ensemble_filepath = Path("models/mnist_ensemble_2")
    distilled_model_filepath = Path("models/distilled_model_dirichlet_test")


    #prob_ensemble = create_ensemble(train_loader, test_loader, args, num_ensemble_members, ensemble_filepath)

    prob_ensemble = ensemble.Ensemble()
    prob_ensemble.load_ensemble(ensemble_filepath)
    LOGGER.info("Ensemble accuracy on test data: {}".format(
        get_accuracy_iter(prob_ensemble, test_loader)))

    class_type = distilled_network.DirichletProbabilityDistribution
    distilled_model = create_distilled_model(train_loader, test_loader, args, prob_ensemble, distilled_model_filepath,
                                             class_type)

    #distilled_model = torch.load(distilled_model_filepath)

    #dirichlet_test(train_loader, test_loader, args, prob_ensemble)
    #effect_of_ensemble_size(prob_ensemble, train_loader, test_loader, args)
    entropy_histogram(prob_ensemble, distilled_model, test_loader)
    test_sample = next(iter(test_loader))
    entropy_comparison_rotation(prob_ensemble, distilled_model, test_sample)
    #noise_effect_on_entropy(distilled_model, prob_ensemble, test_loader)

if __name__ == "__main__":
    main()
