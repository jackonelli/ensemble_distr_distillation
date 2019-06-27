import numpy as np


def accuracy(self, data, model):
    model.eval()  # Räcker detta för att jag ska få ut en nd_array?
    output = model.forward(data.x)
    predictions = np.argmax(output)

    acc = (1 / data.x.shape[0]) * np.sum(predictions == data.y)
    return acc


def effect_of_ensemble_size():
    pass


def effect_of_model_capacity():
    pass


