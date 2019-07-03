import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
import distilled_network
import ensemble
import experiments
import metrics
import models
import utils


def generate_rotated_data_set(img, angles):
    data_set = [nn.functional.rotate(img, angle=angle) for angle in angles]

    return data_set


def get_accuracy(model, data_loader):

    accuracy = 0
    num_batches = 0
    for batch in data_loader:
        inputs, labels = batch
        accuracy += experiments.calculate_accuracy(model, inputs, labels)
        num_batches += 1

    return accuracy / num_batches


def load_mnist_data(batch_size_train=32, batch_size_test=32):

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '/files/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                utils.ReshapeTransform((-1, ))
            ])),  # torchvision.transforms.Normalize((0.1307,), (0.3081,)
        batch_size=batch_size_train,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        '/files/',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            utils.ReshapeTransform((-1, ))
        ])),
                                              batch_size=batch_size_test,
                                              shuffle=True)

    return train_loader, test_loader


def plot_data_set(data_set):
    fig, axes = plt.subplots(2, 5)
    axes[0, 0].imshow(data_set[0, :, :])
    axes[0, 1].imshow(data_set[1, :, :])
    axes[0, 2].imshow(data_set[2, :, :])
    axes[0, 3].imshow(data_set[3, :, :])
    axes[0, 4].imshow(data_set[4, :, :])
    axes[1, 0].imshow(data_set[5, :, :])
    axes[1, 1].imshow(data_set[6, :, :])
    axes[1, 2].imshow(data_set[7, :, :])
    axes[1, 3].imshow(data_set[8, :, :])
    axes[1, 4].imshow(data_set[9, :, :])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def rotation_entropy(img, angles, model):
    data_set = generate_rotated_data_set(img, angles)

    plot_data_set(data_set)

    data_set_flattened = data_set.view(data_set.shape[0],
                                       data_set.shape[1] * data_set.shape[2])

    output = model.predict(data_set_flattened)
    entropy = metrics.entropy(output)

    prediction = torch.argmax(output, dim=-1)

    return entropy, prediction


def main():
    args = utils.parse_args()

    train_loader, test_loader = load_mnist_data()

    input_size = 784
    hidden_size_1 = 54
    hidden_size_2 = 32
    output_size = 10

    prob_ensemble = ensemble.Ensemble()
    num_ensemble_members = 1
    for i in range(num_ensemble_members):
        print("Training ensemble member number {}".format(i + 1))
        model = models.NeuralNet(input_size,
                                 hidden_size_1,
                                 hidden_size_2,
                                 output_size,
                                 lr=args.lr)
        model.train(train_loader, args.num_epochs)
        print("Accuracy on test data: {}".format(
            get_accuracy(model, test_loader)))
        prob_ensemble.add_member(model)

    print("Ensemble accuracy on test data {}".format(
        get_accuracy(prob_ensemble, test_loader)))

    distilled_model = distilled_network.PlainProbabilityDistribution(
        input_size,
        hidden_size_1,
        hidden_size_2,
        output_size,
        prob_ensemble,
        lr=0.001)

    print("Training distilled network")
    distilled_model.train(train_loader, args.num_epochs * 2, t=1)
    print("Distilled model accuracy on test data: {}".format(
        get_accuracy(distilled_model, test_loader)))

    num_points = 10
    angles = torch.tensor(-np.pi / 2, np.pi / 2, np.pi / num_points)

    #test_img =

    ensemble_rotation_entropy, ensemble_rotation_prediction = rotation_entropy(
        test_img, angles, prob_ensemble)
    distilled_model_entropy, distilled_model_rotation_prediction = rotation_entropy(
        test_img, angles, distilled_model)

    plt.plot(angles, ensemble_rotation_entropy)
    plt.plot(angles, distilled_model_entropy)
    plt.xlabel('Rotation angle')
    plt.ylabel('Entropy')
    plt.legend(["Ensemble", "Distilled model"])
    plt.legend()

    print(ensemble_rotation_prediction)
    print(distilled_model_rotation_prediction)


if __name__ == "__main__":
    main()
