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
from src.distilled import logits_prob_dist_subnet
from src.distilled import dummy_logits_probability_distribution
from src.distilled import logits_matching
from src.distilled import cross_entropy_soft_labels
from src.distilled import softmax_matching
from src.distilled import vanilla_distill

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
                                 learning_rate=args.lr*0.01,
                                 scale_teacher_logits=True)

    loss_metric = metrics.Metric(name="Mean loss", function=distilled_model.calculate_loss)
    #softmax_se_metric = metrics.Metric(name="Softmax SE", function=distilled_model.softmax_rmse) MÅSTE FIXA DENNA FÖR DEN FUNGERAR INTE NÄR VI HAR EN VARIANSPAR OCKSÅ
    #softmax_ce_metric = metrics.Metric(name="Softmax CE", function=distilled_model.softmax_xentropy)
    #acc_metric = metrics.Metric(name="Accuracy", function=metrics.accuracy_logits)
    mean_metric = metrics.Metric(name="Mean expected value", function=distilled_model.mean_expected_value)  # Bör generalisera detta sen
    var_metric = metrics.Metric(name="Mean variance", function=distilled_model.mean_variance)
    distilled_model.add_metric(loss_metric)
    distilled_model.add_metric(mean_metric)
    distilled_model.add_metric(var_metric)

#    distilled_model = torch.load(filepath)
    distilled_model.train(train_loader, num_epochs=100)  #args.num_epochs, validation_loader
    #torch.save(distilled_model, filepath)

    #LOGGER.info("Distilled model accuracy on test data: {}".format(
    #    get_accuracy(distilled_model, test_loader)))

    return distilled_model


def create_ensemble(train_loader, validation_loader, test_loader, args, num_ensemble_members,
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
                                                    learning_rate=args.lr*10)

        prob_ensemble.add_member(member)

    loss_metric = metrics.Metric(name="Loss", function=member.calculate_loss)
    prob_ensemble.add_metrics([loss_metric])
    prob_ensemble.train(train_loader, validation_loader=validation_loader, num_epochs=5)
    #LOGGER.info("Ensemble accuracy on test data: {}".format(
    #    get_accuracy(prob_ensemble, test_loader)))

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

    ensemble_rotation_prediction = prob_ensemble.predict(rotated_data_set)
    ensemble_rotation_entropy = metrics.entropy(None, ensemble_rotation_prediction)

    ensemble_member_rotation_prediction = ensemble_member.predict(rotated_data_set)
    ensemble_member_rotation_entropy = metrics.entropy(None, ensemble_member_rotation_prediction)

    distilled_model_rotation_prediction = distilled_model.predict(rotated_data_set)
    distilled_model_rotation_entropy = metrics.entropy(None, distilled_model_rotation_prediction)

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


def noise_effect_on_entropy(distilled_model, data_loader):
    """Effect on entropy of ensemble and model with increasing noise added to the input"""

    inputs, _ = next(iter(data_loader))

    prob_ensemble = distilled_model.teacher

    epsilon = torch.linspace(0.0001, 1, 10)
    ensemble_entropy = torch.zeros([
        len(epsilon),
    ])
    ensemble_member_entropy = torch.zeros([
        len(epsilon), len(prob_ensemble.members)
    ])
    distilled_model_entropy = torch.zeros([
        len(epsilon),
    ])

    for i, e in enumerate(epsilon):

        distr = torch.distributions.Normal(0, e)
        input_perturbed = inputs + distr.sample(
            (inputs.shape[0], inputs.shape[1]))

        ensemble_prediction = torch.mean(prob_ensemble.predict(input_perturbed), dim=1)
        ensemble_entropy[i] += torch.mean(metrics.entropy(ensemble_prediction, None))

        for j in range(len(prob_ensemble.members)):
            ensemble_member_prediction = prob_ensemble.members[j].predict(input_perturbed)
            ensemble_member_entropy[i, j] += torch.mean(metrics.entropy(ensemble_member_prediction, None))

        distilled_model_prediction = distilled_model.predict(input_perturbed)
        distilled_model_prediction = torch.mean(torch.cat((distilled_model_prediction,
                                                1- torch.sum(distilled_model_prediction, dim=-1, keepdim=True)),
                                                dim=-1), dim=1)

        distilled_model_entropy[i] += torch.mean(metrics.entropy(distilled_model_prediction, None, correct_nan=True))

        if i == (len(epsilon) - 1):
            # We will investigate the results on the noisiest data a bit more
            # We want to look at the mean logits for all x, ensemble and distilled model
            teacher_logits = distilled_model._generate_teacher_predictions(input_perturbed)
            teacher_mean_logits = torch.mean(teacher_logits, dim=[0, 1])
            teacher_mean_variance = torch.mean(torch.var(teacher_logits, dim=1), dim=0)
            student_mean_logits = torch.mean(distilled_model.predict_logits(input_perturbed), dim=[0, 1])
            student_mean_variance = torch.mean(distilled_model.forward(input_perturbed)[1], dim=0)

    print('Ensemble mean logits: {}'.format(teacher_mean_logits))
    print('Distilled model mean logits: {}'.format(student_mean_logits))
    print('Ensemble mean variance: {}'.format(teacher_mean_variance))
    print('Distilled model mean variance: {}'.format(student_mean_variance))

    epsilon = epsilon.data.numpy()
    plt.plot(epsilon, ensemble_entropy.data.numpy(), label='Ensemble')

    for j in range(len(prob_ensemble.members)):
        plt.plot(epsilon, ensemble_member_entropy[:, j].data.numpy(), label='Ensemble member ' + str(j))

    plt.plot(epsilon, distilled_model_entropy.data.numpy(), label='Distilled model')
    plt.xlabel('\u03B5')
    plt.ylabel('Entropy')
    plt.legend()
    plt.show()


def effect_of_ensemble_size(full_ensemble, train_loader, test_loader, args):
    """Effect of ensemble size on error and nll of distilled model"""

    ensemble_size = len(full_ensemble.members)

    ensemble_error = torch.zeros([
        ensemble_size,
    ])
    distilled_model_error = torch.zeros([
        ensemble_size,
    ])

    ensemble_nll = torch.zeros([
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
        distilled_model = create_distilled_model(train_loader, test_loader, args, concat_ensemble, filepath)

        ensemble_error[i], distilled_model_error[i] = 1 - get_accuracy(distilled_model, test_loader)
        ensemble_nll[i], distilled_model_nll[i] = 1 - get_entropy(distilled_model, test_loader)

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(np.arange(ensemble_size), ensemble_error.data.numpy())
    axes[0].plot(np.arange(ensemble_size), distilled_model_error.data.numpy())
    axes[0].set_xlabel('Ensemble size')
    axes[0].set_ylabel('Error')
    axes[0].legend(['Ensemble', 'Distilled model'])

    axes[1].plot(np.arange(ensemble_size), ensemble_nll.data.numpy())
    axes[1].plot(np.arange(ensemble_size), distilled_model_nll.data.numpy())
    axes[1].set_xlabel('Ensemble size')
    axes[1].set_ylabel('Entropy')
    axes[1].legend(['Ensemble', 'Distilled model'])
    plt.show()


def get_histogram(distilled_model, test_loader, obj_fun, label):
    """Comparison of entropy histograms of ensemble and model"""

    ensemble_metric, distilled_model_metric = obj_fun(distilled_model, test_loader)

    num_bins = 100

    fig, axes = plt.subplots(np.int(np.ceil(distilled_model_metric.shape[-1]/2)), 2)

    for i, ax in enumerate(axes.reshape(-1)):
        if i < distilled_model_metric.shape[-1]:
            ax.hist(ensemble_metric[:, i], bins=num_bins, alpha=0.5, density=True)
            ax.hist(distilled_model_metric[:, i], bins=num_bins, alpha=0.5, density=True)
            ax.set_xlabel(label)
            ax.legend(['Ensemble', 'Distilled model'])

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


def get_accuracy(distilled_model, data_loader, label='test'):
    # SKA KANSKE INTE VARA EN EGEN METOD; EV. SKICKA IN METRIC
    prob_ensemble = distilled_model.teacher

    inputs, labels = next(iter(data_loader))

    teacher_distribution = torch.mean(prob_ensemble.predict(inputs), dim=1)
    teacher_acc = metrics.accuracy(teacher_distribution, labels)
    LOGGER.info("Ensemble model accuracy on {} data {}".format(label, teacher_acc))

    student_distribution = torch.mean(distilled_model.predict(inputs), dim=1)
    student_distribution = torch.cat((student_distribution,
         1 - torch.sum(student_distribution, dim=-1, keepdim=True)), dim=-1)
    student_acc = metrics.accuracy(student_distribution, labels)
    LOGGER.info("Distilled model accuracy on {} data {}".format(label, student_acc))

    # Find student predictions relative teacher predictions
    teacher_labels, _ = utils.tensor_argmax(teacher_distribution)
    student_acc_teacher = metrics.accuracy(student_distribution, teacher_labels.int())
    LOGGER.info("Distilled model accuracy on {} data relative teacher {}".format(label, student_acc_teacher))

    return teacher_acc, student_acc


def get_entropy(distilled_model, data_loader, label='test'):
    # KAN EVENTUELLT HA EN GEMENSAM METOD MED ACCURACY OCH GÖRA EN NY METRIC, KAN LOOPA ÖVER TUPELSEN
    prob_ensemble = distilled_model.teacher

    inputs, labels = next(iter(data_loader))

    teacher_distribution = prob_ensemble.predict(inputs)
    ensemble_entropy = metrics.uncertainty_separation_entropy(teacher_distribution, None)
    ensemble_entropy = np.stack((ensemble_entropy[0].data.numpy(), ensemble_entropy[1].data.numpy(),
                                 ensemble_entropy[2].data.numpy()), axis=1)

    student_distribution = distilled_model.predict_logits(inputs)
    student_distribution = torch.cat((student_distribution,
                                      torch.ones([inputs.shape[0], student_distribution.shape[1], 1])), dim=-1)
    distilled_model_entropy = metrics.uncertainty_separation_entropy(student_distribution, None, logits=True)
    distilled_model_entropy = np.stack((distilled_model_entropy[0].data.numpy(),
                                        distilled_model_entropy[1].data.numpy(),
                                        distilled_model_entropy[2].data.numpy()), axis=1)

    LOGGER.info("Ensemble mean total entropy on {} data {}".format(label, np.mean(ensemble_entropy[:, 0])))
    LOGGER.info("Ensemble mean epistemic entropy on {} data {}".format(label, np.mean(ensemble_entropy[:, 1])))
    LOGGER.info("Ensemble mean aleatoric entropy on {} data {}".format(label, np.mean(ensemble_entropy[:, 2])))

    LOGGER.info("Distilled model mean total entropy on {} data {}".format(label, np.mean(distilled_model_entropy[:, 0])))
    LOGGER.info("Distilled model mean epistemic entropy on {} data {}".format(label, np.mean(distilled_model_entropy[:, 1])))
    LOGGER.info("Distilled model mean aleatoric entropy on {} data {}".format(label, np.mean(distilled_model_entropy[:, 2])))

    return ensemble_entropy, distilled_model_entropy


def get_var(distilled_model, data_loader, label='test'):

    inputs, _ = next(iter(data_loader))

    prob_ensemble = distilled_model.teacher
    ensemble_logits = prob_ensemble.get_logits(inputs).detach().numpy()
    scaled_logits = ensemble_logits - ensemble_logits[:, :, -1][:, :, np.newaxis]
    ensemble_var = np.var(scaled_logits, axis=1)
    LOGGER.info("Min ensemble variance on {} data: {}".format(label, ensemble_var.min()))
    LOGGER.info("Max ensemble variance on {} data: {}".format(label, ensemble_var.max()))
    LOGGER.info("Mean ensemble variance on {} data: {}".format(label, np.mean(ensemble_var, axis=0)))

    # We will also look at the variance over all samples
    scaled_logits = scaled_logits.reshape(scaled_logits.shape[0] * scaled_logits.shape[1], scaled_logits.shape[2])
    LOGGER.info("Total mean ensemble variance on {} data: {}".format(label, np.var(scaled_logits, axis=0)))

    distilled_model_var = distilled_model.forward(inputs)[1].detach().numpy()
    LOGGER.info("Min distilled model variance on {} data: {}".format(label, distilled_model_var.min()))
    LOGGER.info("Max distilled model variance on {} data: {}".format(label, distilled_model_var.max()))
    LOGGER.info("Mean distilled model variance on {} data: {}".format(label, np.mean(distilled_model_var, axis=0)))

    return ensemble_var, distilled_model_var


def plot_metrics(model):
    fig, axes = plt.subplots(2, 2)
    for i, metric in enumerate(model.metrics.values()):

        if isinstance(metric.memory[1], torch.FloatTensor):
            metric_batch_mean = torch.stack(metric.memory[1:]).data.numpy()
        else:
            metric_batch_mean = np.stack(metric.memory[1:])
        axes.reshape(-1)[i].plot(np.arange(metric_batch_mean.shape[0]), metric_batch_mean)
        axes.reshape(-1)[i].set_ylabel(metric.name)
        axes.reshape(-1)[i].set_xlabel('it')

    plt.show()


def distillation(class_type, distilled_output_dim, ensemble_filepath, distilled_model_filepath, train_ensemble=False):
    args = utils.parse_args()

    log_file = Path("{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    utils.setup_logger(log_path=Path.cwd() / args.log_dir / log_file,
                       log_level=args.log_level)

    train_set = mnist.MnistData()
    valid_set = mnist.MnistData(data_set='validation')
    test_set = mnist.MnistData(train=False)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    train_loader_full = torch.utils.data.DataLoader(train_set,
                                                    batch_size=len(train_set),
                                                    shuffle=True,
                                                    num_workers=0)

    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=0)

    valid_loader_full = torch.utils.data.DataLoader(valid_set,
                                                    batch_size=len(valid_set),
                                                    shuffle=True,
                                                    num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=10000,
                                              shuffle=True,
                                              num_workers=0)

    num_ensemble_members = 10
    if train_ensemble:
        prob_ensemble = create_ensemble(train_loader, valid_loader, test_loader, args, num_ensemble_members,
                                        ensemble_filepath)
    else:
        prob_ensemble = ensemble.Ensemble(num_ensemble_members)
        prob_ensemble.load_ensemble(ensemble_filepath, num_members=num_ensemble_members)


    # prob_ensemble.calc_metrics(test_loader)

    #distilled_model = create_distilled_model(distilled_output_dim, train_loader, valid_loader, test_loader, args,
    #                                         prob_ensemble, distilled_model_filepath, class_type)

    distilled_model = torch.load(distilled_model_filepath)

    #get_var(distilled_model, train_loader_full, 'train')
    #get_var(distilled_model, valid_loader_full, 'validation')
    #get_var(distilled_model, test_loader)

    #get_accuracy(distilled_model, test_loader)
    #get_accuracy(distilled_model, train_loader_full, label='train')

    get_entropy(distilled_model, test_loader)

    #plot_metrics(distilled_model)
    # distilled_model = torch.load(distilled_model_filepath)

    # effect_of_ensemble_size(prob_ensemble, train_loader, test_loader, args)
    #get_histogram(distilled_model, test_loader, get_var, 'Variance')
    get_histogram(distilled_model, train_loader_full, get_entropy, 'Entropy')
    noise_effect_on_entropy(distilled_model, test_loader)

    # test_sample = test_set.get_sample(5)
    # entropy_comparison_rotation(prob_ensemble, distilled_model, test_sample)
    # noise_effect_on_entropy(distilled_model, prob_ensemble, test_loader)


def main():
    """Main"""

    # Distribution distillation
    class_type = logits_probability_distribution.LogitsProbabilityDistribution
    output_dim = 18
    distilled_model_filepath = Path("models/distilled_mnist_logits_model")

    # Distribution distillation, subnetworks
    #class_type = logits_prob_dist_subnet.LogitsProbabilityDistributionSubNet
    #output_dim = 18
    #distilled_model_filepath = Path("models/distilled_mnist_logits_model_subnets")

    # Dummy distillation
    #class_type = dummy_logits_probability_distribution.DummyLogitsProbabilityDistribution
    #output_dim = 18
#    distilled_model_filepath = Path("models/distilled_mnist_dummy_logits_model")


    # Logits matching
    #class_type = logits_matching.LogitsMatching
    #output_dim = 9
#    distilled_model_filepath = Path("models/distilled_mnist_logits_matching")



    # Softmax matching
    #class_type = softmax_matching.SoftmaxMatching
    #output_dim = 9
#    distilled_model_filepath = Path("models/distilled_softmax_matching")


    # Cross entropy with soft labels
    #class_type = cross_entropy_soft_labels.XCSoftLabels
    #output_dim = 9

    # Vanilla distillation
    #class_type = vanilla_distill.VanillaDistill
    #output_dim = 9
#    distilled_model_filepath = Path("models/distilled_mnist_vanilla_model")

    ensemble_filepath = Path("models/mnist_ensemble_10")
    distillation(class_type, output_dim, ensemble_filepath, distilled_model_filepath)
    # OBS HAR KOMMENTERAT BORT SPARNINGEN


if __name__ == "__main__":
    main()
