"""Test of mnist ensemble and distilled model"""

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
from src.distilled import logits_probability_distribution
from src.distilled import dummy_logits_probability_distribution
from src.distilled import logits_matching
from src.distilled import cross_entropy_soft_labels
from src.distilled import softmax_matching

import src.loss as custom_loss

LOGGER = logging.getLogger(__name__)


def create_distilled_model(output_size,
                           train_loader,
                           valid_loader,
                           test_loader,
                           args,
                           prob_ensemble,
                           filepath,
                           class_type=dirichlet_probability_distribution.
                           DirichletProbabilityDistribution):
    """Create a distilled network trained with ensemble output"""

    input_size = 784
    hidden_size_1 = 54
    hidden_size_2 = 32

    distilled_model = class_type(input_size,
                                 hidden_size_1,
                                 hidden_size_2,
                                 output_size,
                                 prob_ensemble,
                                 learning_rate=args.lr)

    loss_metric = metrics.Metric(name="Mean val loss", function=distilled_model.calculate_loss)
    #loss_norm_metric = metrics.Metric(name="Loss normalizer", function=custom_loss.gaussian_neg_log_likelihood_normalizer)
    #loss_ll_metric = metrics.Metric(name="Loss ll", function=custom_loss.gaussian_neg_log_likelihood_ll)
    distilled_model.add_metric(loss_metric)
    #distilled_model.add_metric(loss_norm_metric)
    #distilled_model.add_metric(loss_ll_metric)

#    distilled_model = torch.load(filepath)
    distilled_model.train(train_loader, validation_loader=valid_loader, num_epochs=50)  #args.num_epochs, validation_loader
    torch.save(distilled_model, filepath)

    #LOGGER.info("Distilled model accuracy on test data: {}".format(
    #    get_accuracy(distilled_model, test_loader)))

    return distilled_model


def create_ensemble(train_loader, test_loader, args, num_ensemble_members,
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
    prob_ensemble.train(train_loader, args.num_epochs)
    LOGGER.info("Ensemble accuracy on test data: {}".format(
        get_accuracy(prob_ensemble, test_loader)))

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
        prob_ensemble, [rotated_data_set])
    ensemble_member_rotation_entropy, ensemble_member_rotation_prediction = \
        get_entropy(ensemble_member, [rotated_data_set])
    distilled_model_rotation_entropy, distilled_model_rotation_prediction = \
        get_entropy(distilled_model, [rotated_data_set])

    LOGGER.info("True label is: {}".format(test_label))
    LOGGER.info("Ensemble prediction: {}".format(ensemble_rotation_prediction))
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


def get_accuracy(model, data_loader):
    """Calculate accuracy of model on data in dataloader"""

    accuracy = 0
    num_batches = 0
    for batch in data_loader:
        inputs, labels = batch
        predicted_distribution = model.predict(inputs)
        accuracy += metrics.accuracy(labels, predicted_distribution)
        num_batches += 1

    return accuracy / num_batches


def get_entropy(model, test_loader):
    """Calculate entropy of model output over a dataloader"""

    entropy = []
    prediction = []
    for i, batch in enumerate(test_loader):
        inputs, labels = batch

        output = model.predict(inputs)
        entropy.append(metrics.entropy(None, output))

        prediction.append(torch.max(output, dim=-1))

    return torch.stack(entropy, dim=0), torch.stack(prediction, dim=0)


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

            ensemble_entropy[i] += torch.sum(
                get_entropy(ensemble, input_perturbed)[0])
            ensemble_member_entropy[i] += torch.sum(
                get_entropy(ensemble_member, input_perturbed)[0])
            model_entropy[i] += torch.sum(
                get_entropy(model, input_perturbed)[0])

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

    ensemble_error = torch.zeros([
        ensemble_size,
    ])
    ensemble_member_error = torch.zeros([
        ensemble_size,
    ])
    distilled_model_error = torch.zeros([
        ensemble_size,
    ])

    ensemble_nll = torch.zeros([
        ensemble_size,
    ])
    ensemble_member_nll = torch.zeros([
        ensemble_size,
    ])
    distilled_model_nll = torch.zeros([
        ensemble_size,
    ])

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

        ensemble_error[i] = 1 - get_accuracy(concat_ensemble, test_loader)
        ensemble_member_error[i] = 1 - get_accuracy(ensemble_member,
                                                    test_loader)
        distilled_model_error[i] = 1 - get_accuracy(distilled_model,
                                                    test_loader)
        ensemble_nll[i] = torch.sum(
            get_entropy(concat_ensemble, test_loader)[0])
        ensemble_member_nll[i] = torch.sum(
            get_entropy(ensemble_member, test_loader)[0])
        distilled_model_nll[i] = torch.sum(
            get_entropy(distilled_model, test_loader)[0])

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

    ensemble_entropy = get_entropy(ensemble, test_loader).data.numpy()
    ensemble_member_entropy = get_entropy(ensemble_member,
                                          test_loader).data.numpy()
    distilled_model_entropy = get_entropy(model, test_loader).data.numpy()

    num_bins = 100
    plt.hist(ensemble_entropy, bins=num_bins, alpha=0.5, density=True)
    plt.hist(ensemble_member_entropy, bins=num_bins, alpha=0.5, density=True)
    plt.hist(distilled_model_entropy, bins=num_bins, alpha=0.5, density=True)
    plt.xlabel('Entropy')
    plt.legend(['Ensemble model', 'Ensemble member', 'Distilled model'])

    plt.show()


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


def get_accuracy(distilled_model, test_loader, label='test'):

    prob_ensemble = distilled_model.teacher

    test_inputs, test_labels = next(iter(test_loader))
    test_labels = test_labels.data.numpy()

    teacher_test_predictions = prob_ensemble.predict(test_inputs)
    teacher_predictions = torch.argmax(torch.mean(teacher_test_predictions, axis=1), axis=-1).data.numpy()
    teacher_acc = np.mean(teacher_predictions == test_labels)
    LOGGER.info("Ensemble accuracy on test data {}".format(teacher_acc))

    student_test_predictions = distilled_model.predict(test_inputs)
    student_predictions = torch.argmax(torch.cat((student_test_predictions, 1-torch.sum(student_test_predictions, dim=1,
                                                                                        keepdim=True)), dim=1),
                                       axis=1).data.numpy()
    student_acc = np.mean(np.transpose(student_predictions) == test_labels)
    LOGGER.info("Distilled model accuracy on {} data {}".format(label, student_acc))


def distillation(class_type, distilled_output_dim):
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
                                              batch_size=10000,
                                              shuffle=True,
                                              num_workers=0)

    train_loader_full = torch.utils.data.DataLoader(train_set,
                                                    batch_size=train_set.n_samples,
                                                    shuffle=True,
                                                    num_workers=0)

    # num_ensemble_members = 10

    ensemble_filepath = Path("models/mnist_ensemble_10")
    distilled_model_filepath = Path("models/distilled_mnist_logits_model_one_member_cross_entropy")

    prob_ensemble = ensemble.Ensemble(output_size=10)
    prob_ensemble.load_ensemble(ensemble_filepath, num_members=1)
    # prob_ensemble.calc_metrics(test_loader)

    ensemble_train_var = np.var(prob_ensemble.get_logits(
        torch.tensor(train_set.data.reshape(train_set.data.shape[0], 28 ** 2) / 255,
                     dtype=torch.float32)).detach().numpy(),
                                axis=1)
    LOGGER.info("Max ensemble variance: {}".format(ensemble_train_var.min()))
    LOGGER.info("Min ensemble variance: {}".format(ensemble_train_var.max()))

    distilled_model = create_distilled_model(distilled_output_dim, train_loader, valid_loader, test_loader, args,
                                            prob_ensemble,
                                            distilled_model_filepath,
                                            class_type)

    #distilled_model = torch.load(distilled_model_filepath)
    for metric in distilled_model.metrics.values():
        metric_batch_mean = torch.stack(metric.memory[1:]).data.numpy()
        plt.plot(np.arange(metric_batch_mean.shape[0]), metric_batch_mean)
        print(metric_batch_mean)

    plt.legend(['nll', 'nll - normalizing constant', 'nll - squared error'])
    plt.show()

    get_accuracy(distilled_model, test_loader)
    get_accuracy(distilled_model, train_loader_full, label='train')


    # distilled_model = torch.load(distilled_model_filepath)

    # effect_of_ensemble_size(prob_ensemble, train_loader, test_loader, args)
    # entropy_histogram(prob_ensemble, distilled_model, test_loader)

    # test_sample = test_set.get_sample(5)
    # entropy_comparison_rotation(prob_ensemble, distilled_model, test_sample)
    # noise_effect_on_entropy(distilled_model, prob_ensemble, test_loader)


def main():
    """Main"""

    # Distillation
    #class_type = logits_probability_distribution.LogitsProbabilityDistribution
    # output_dim = 18

    # Dummy distillation
    #class_type = dummy_logits_probability_distribution.DummyLogitsProbabilityDistribution
    #output_dim = 18

    # Logits matching
    #class_type = logits_matching.LogitsMatching
    #output_dim = 9


    # Softmax matching
    class_type = softmax_matching.SoftmaxMatching
    output_dim = 10

    # Cross entropy with soft labels
    #class_type = cross_entropy_soft_labels.XCSoftLabels
    #output_dim = 10

    distillation(class_type, output_dim)


if __name__ == "__main__":
    main()
