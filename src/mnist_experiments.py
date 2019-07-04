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
import experiments
import metrics
import models
import utils


LOGGER = logging.getLogger(__name__)


def generate_rotated_data_set(img, angles):
    img = torchvision.transforms.ToPILImage()(img)
    data_set = [torch.squeeze(torchvision.transforms.ToTensor()(torchvision.transforms.functional.rotate(img, angle=angle)))
                for angle in angles]

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

    mnist_transform = torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),# torchvision.transforms.Normalize((0.1307,), (0.3081,)
                                       utils.ReshapeTransform((-1,))])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=mnist_transform),
        batch_size=batch_size_train,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=mnist_transform),
        batch_size=batch_size_test,
        shuffle=True)

    return train_loader, test_loader


def plot_data_set(data_set):
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


def rotation_entropy(img, angles, model):
    data_set = generate_rotated_data_set(img, angles)

    #plot_data_set(data_set)

    data_set_flattened = torch.stack([data_point.view(28 * 28) for data_point in data_set])

    output = model.predict(data_set_flattened)
    entropy = metrics.entropy(output)

    prediction = torch.argmax(output, dim=-1)

    return entropy, prediction


def create_ensemble(train_loader, test_loader, num_ensemble_members, args):

    input_size = 784
    hidden_size_1 = 54
    hidden_size_2 = 32
    output_size = 10

    prob_ensemble = ensemble.Ensemble()

    filepaths = []
    for i in range(num_ensemble_members):
        LOGGER.info("Training ensemble member number {}".format(i + 1))
        model = models.NeuralNet(input_size, hidden_size_1, hidden_size_2, output_size, learning_rate=args.lr)
        model.train(train_loader, args.num_epochs)
        LOGGER.info("Accuracy on test data: {}".format(get_accuracy(model, test_loader)))
        prob_ensemble.add_member(model)
        filepaths.append(Path("models/ensemble_member_" + str(i+5)))

    LOGGER.info("Ensemble accuracy on test data: {}".format(get_accuracy(prob_ensemble, test_loader)))

    prob_ensemble.save_ensemble(filepaths)

    return prob_ensemble


def create_distilled_model(train_loader, test_loader, ensemble, args):

    input_size = 784
    hidden_size_1 = 54
    hidden_size_2 = 32
    output_size = 10

    distilled_model = distilled_network.PlainProbabilityDistribution(
        input_size, hidden_size_1, hidden_size_2, output_size, ensemble, learning_rate=args.lr*0.1)

    distilled_model.train(train_loader, args.num_epochs * 2, t=1)
    LOGGER.info("Distilled model accuracy on test data: {}".format(get_accuracy(distilled_model, test_loader)))
    torch.save(distilled_model, Path("data/distilled_model"))

    return distilled_model


def load_ensemble(num_ensemble_members):

    #Finns med stor sannolikhet något snyggare sätt att göra detta på
    filepaths = []
    for i in range(num_ensemble_members):
        filepaths.append(Path("models/ensemble_member_" + str(i)))

    prob_ensemble = ensemble.Ensemble()
    prob_ensemble.load_ensemble(filepaths)

    return prob_ensemble


def main():

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_loader, test_loader = load_mnist_data()

    num_ensemble_members = 5
    prob_ensemble = create_ensemble(train_loader, test_loader, num_ensemble_members, args)
    #     distilled_model = torch.load(Path("data/distilled_model"))

    distilled_model = create_distilled_model(train_loader, test_loader, prob_ensemble, args)

    num_points = 10
    max_val = 90
    angles = torch.arange(-max_val, max_val, max_val * 2 / num_points)

    test_sample = next(iter(test_loader))
    test_img = test_sample[0][0].view(28, 28)
    test_label = test_sample[1][0]

    ensemble_member = prob_ensemble.members[0]
    ensemble_rotation_entropy, ensemble_rotation_prediction = rotation_entropy(test_img, angles, prob_ensemble)
    ensemble_member_rotation_entropy, ensemble_member_rotation_prediction = \
        rotation_entropy(test_img, angles, ensemble_member)
    distilled_model_rotation_entropy, distilled_model_rotation_prediction = \
        rotation_entropy(test_img, angles, distilled_model)

    angles = angles.data.numpy()
    plt.plot(angles, ensemble_rotation_entropy.data.numpy(), 'o--')
    plt.plot(angles, ensemble_member_rotation_entropy.data.numpy(), 'o-')
    plt.plot(angles, distilled_model_rotation_entropy.data.numpy(), 'o-')

    plt.xlabel('Rotation angle')
    plt.ylabel('Entropy')
    plt.legend(["Ensemble", "Ensemble member", "Distilled model"])
    plt.show()

    print("True label is: {}".format(test_label))
    print(ensemble_rotation_prediction.data.numpy())
    print(ensemble_member_rotation_prediction.data.numpy())
    print(distilled_model_rotation_prediction.data.numpy())


if __name__ == "__main__":
    main()
