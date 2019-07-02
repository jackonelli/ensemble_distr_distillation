""""Main entry point"""
from pathlib import Path
import numpy as np
import torch
from dataloaders import gaussian
import utils
import models
import distilled_network
import ensemble
import experiments


def main():
    """Main"""
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
        print("Training ensemble member number {}".format(i+1))
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

    ensemble_accuracy, model_accuracy = experiments.accuracy_comparison(distilled_model, prob_ensemble, inputs, labels)
    print(ensemble_accuracy)
    print(model_accuracy)

    #ensemble_nll, model_nll = experiments.nll_comparison(distilled_model, prob_ensemble, inputs, labels, 2)
    #print(ensemble_nll)
    #print(model_nll)

    experiments.entropy_comparison_plot(distilled_model, prob_ensemble, inputs)


if __name__ == "__main__":
    main()
