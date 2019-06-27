import numpy as np
import matplotlib.pyplot as plt


def accuracy_comparison(model, ensemble, data):
    ensemble.eval_mode()
    ensemble_output = ensemble.prediction()
    ensemble_prediction = np.argmax(ensemble_output, axis=-1)
    ensemble_accuracy = (1 / data.x.shape[0]) * np.sum(ensemble_prediction == data.y)

    model.eval()  # Räcker detta för att jag ska få ut en nd_array? Eller måste jag ha en extra eval-funktion i nätverket?
    model_output = model.forward(data.x)
    model_prediction = np.argmax(model_output)
    model_accuracy = (1 / data.x.shape[0]) * np.sum(model_prediction == data.y)

    return ensemble_accuracy, model_accuracy


def brier_score_comparison(model, ensemble, data):  # Eventuellt
    pass

def effect_of_ensemble_size():
    pass


def effect_of_model_capacity():
    pass


def entropy(p):
    return - p * np.log(p)


def entropy_comparison(model, ensemble, data):
    # Comparing predictions vs entropy of ensemble and distilled model

    ensemble.eval_mode()
    ensemble_output = ensemble.prediction()
    ensemble_entropy = (1 / data.x.shape[0]) * np.sum(entropy(ensemble_output), axis=-1)

    model.eval()
    model_output = model.forward(data.x)
    model_entropy = (1 / data.x.shape[0]) * np.sum(entropy(model_output), axis=-1)  # Logiskt att kolla på detta värde?

    num_bins = 100
    plt.hist(ensemble_entropy, bins=num_bins, density=True)
    plt.hist(model_entropy, bins=num_bins, density=True)
    plt.xlabel('Entropy')
    plt.legend(['Ensemble model', 'Distilled model'])
    plt.show()


def to_one_hot(y):
    num_classes = np.max(y) + 1
    # if num_classes > 2
    return np.eye(num_classes)[y]


def nll_comparison(model, ensemble, data):
    ensemble.eval_mode()
    ensemble_output = ensemble.prediction()
    ensemble_nll = np.sum(-to_one_hot(data.y) * np.log(ensemble_output))

    model.eval()
    model_output = model.forward(data.x)
    model_nll = np.sum(-to_one_hot(data.y) * np.log(model_output))

    return ensemble_nll, model_nll


# Sen lite andra allmänna osäkerhetstest?
# Typ i Sensoy et al. så kollar de på entropi + accuracy när de lägger på brus på

def noise_effect_on_entropy(model, ensemble, data):
    # Oklart om det här såhär de gör, men
    epsilon = np.linspace(0.0001, 1, 10)

    ensemble_entropy = np.zeros([len(epsilon), ])
    model_entropy = np.zeros([len(epsilon), ])
    for i, e in enumerate(epsilon):
        data_perturbed = data.copy()
        data_perturbed.x = data_perturbed.x + np.random.normal(loc=0, scale=epsilon, size=data.shape)
        ensemble_entropy[i], model_entropy[i] = entropy_comparison(model, ensemble, data_perturbed)

    plt.plot(epsilon, ensemble_entropy)
    plt.plot(epsilon, model_entropy)
    plt.xlabel('Epsilon')
    plt.ylabel('Entropy')
    plt.legend(['Ensemble model', 'Distilled model'])
    plt.show()


def ood_test(ood_data):
    # What happens with accuracy, entropy etc.?
    pass
