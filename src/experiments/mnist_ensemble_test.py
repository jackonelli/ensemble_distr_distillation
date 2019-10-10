"""Test of mnist ensemble uncertainty"""

import numpy as np
import torch
import torchvision
import logging
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime
from src.distilled import dirichlet_probability_distribution
from src.ensemble import ensemble
import src.metrics as metrics
import src.utils as utils
from src.dataloaders import mnist
from src.ensemble import simple_classifier
import src.utils as utils

LOGGER = logging.getLogger(__name__)


def create_ensemble(train_loader, valid_loader, test_loader, args, num_ensemble_members,
                    filepath):
    """Create an ensemble model"""

    input_size = 784
    hidden_size_1 = 54
    hidden_size_2 = 32
    output_size = 10

    prob_ensemble = ensemble.Ensemble(output_size)

    for i in range(num_ensemble_members):
        LOGGER.info("Training ensemble member number {}".format(i + 1))
        member = simple_classifier.SimpleClassifier(input_size,
                                                    hidden_size_1,
                                                    hidden_size_2,
                                                    output_size,
                                                    learning_rate=args.lr)

        prob_ensemble.add_member(member)

    acc_metric = metrics.Metric(name="Acc", function=metrics.accuracy)
    prob_ensemble.add_metrics([acc_metric])
    prob_ensemble.train(train_loader, int(args.num_epochs), valid_loader)
    prob_ensemble.calc_metrics(test_loader)
    prob_ensemble.save_ensemble(filepath)

    return prob_ensemble


def uncertainty_rotation(model, test_sample):
    """Get uncertainty separation for model on rotated data set"""

    test_img = test_sample[0].view(28, 28)
    test_label = test_sample[1]

    num_points = 100
    max_val = 90
    angles = torch.arange(-max_val, max_val, max_val * 2 / num_points)

    rotated_data_set = generate_rotated_data_set(test_img, angles)

    angle_samples = torch.arange(-max_val, max_val, max_val * 2 / 10)
    small_rotated_data_set = generate_rotated_data_set(test_img, angle_samples)
    plot_data_set(small_rotated_data_set)

    rotated_data_set = torch.stack(
        [data_point.view(28 * 28) for data_point in rotated_data_set])

    predicted_distribution = model.predict(rotated_data_set)
    tot_unc, ep_unc, al_unc = metrics.uncertainty_separation_entropy(predicted_distribution, None)

    LOGGER.info("True label is: {}".format(test_label))
    LOGGER.info("Model prediction: {}".format(np.argmax(predicted_distribution.detach().numpy(), axis=-1)))

    angles = angles.data.numpy()
    plt.plot(angles, tot_unc.data.numpy(), 'o--')
    plt.plot(angles, ep_unc.data.numpy(), 'o--')
    plt.plot(angles, al_unc.data.numpy(), 'o--')
    plt.xlabel('Rotation angle')
    plt.ylabel('Uncertainty')
    plt.legend(["Total", "Epistemic", "Aleatoric"])
    plt.show()


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


def plot_data_set(data_set):
    """Plot of image data set
    Args:
        data_set (list(len=10)): list of ten images/2D ndarrays
    """

    fig, axes = plt.subplots(2, 5)
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(data_set[i])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def main():
    """Main"""

    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = mnist.MnistData()
    valid_set = mnist.MnistData(data_set='validation')
    test_set = mnist.MnistData(train=False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=0)

    num_ensemble_members = 10

    ensemble_filepath = Path("models/mnist_ensemble_10")
    output_size = 10
    prob_ensemble = ensemble.Ensemble(output_size)
    #prob_ensemble.load_ensemble(ensemble_filepath)

    #rob_ensemble = create_ensemble(train_loader, valid_loader, test_loader,
    #                                args, num_ensemble_members, ensemble_filepath)

    prob_ensemble.load_ensemble(ensemble_filepath)

    test_sample = test_set.get_sample(5)
    uncertainty_rotation(prob_ensemble, test_sample)


if __name__ == "__main__":
    main()
