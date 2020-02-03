import logging
import tensorflow as tf
from scipy.special import softmax as scipy_softmax
import src.experiments.cifar10.uq_benchmark_2019.models_lib as models_lib

"""This is a wrapper class that extracts saved data from a specific type of dataset"""


class TensorflowEnsemble:
    def __init__(self, output_size, indices=None):
        """The ensemble member needs to track the size
        of the output of the ensemble
        This can be automatically inferred but it would look ugly
        and this now works as a sanity check as well
        """
        self.members = list()
        self._log = logging.getLogger(self.__class__.__name__)
        self.output_size = output_size
        self.size = 0
        self.indices = indices

    def add_member(self, new_member):
            self._log.info("Adding {} to ensemble".format(type(new_member)))
            self.members.append(new_member)
            self.size += 1

    def add_multiple(self, number_of, constructor):
        for _ in range(number_of):
            self.add_member(constructor())

    def train(self, train_loader, num_epochs, validation_loader=None):
        # We will not have this option for now
        pass

    def get_predictions(self, inputs):
        # Inputs should be a tuple (x, ensemble predictions, ensemble logits)

        predictions = inputs[1]

        if self.indices is not None:
            predictions = predictions[:, self.indices, :]

        return predictions

    def get_logits(self, inputs):
        # Inputs should be a tuple (x, ensemble predictions, ensemble logits)

        logits = inputs[2]

        if self.indices is not None:
            logits = logits[:, self.indices, :]

        return logits

    def predict(self, inputs):
        """Ensemble prediction
        Returns the predictions of all individual ensemble members.
        The return is actually a tuple with (pred_mean, all_predictions)
        for backwards compatibility but this should be removed.
        B = batch size, K = num output params, N = ensemble size


        Args:
            inputs (torch.tensor((B, data_dim))): data batch

        Returns:
            predictions (torch.tensor((B, N, K)))
        """
        logits = []
        for member_ind, member in enumerate(self.members):
            logits.append(member.predict(inputs))

        predictions = tf.stack([scipy_softmax(x, axis=-1) for x in logits], axis=1)
        logits = tf.stack(logits, axis=1)

        return logits, predictions

    def load_ensemble(self, models_dir, num_members=None):

        for i in range(num_members):
            member = models_lib.load_model(models_dir + str(i))
            # TODO: kolla detta för tf-modellerna, DETTA ÄR JU VIKTIGT ÄNDÅ!!! MEN det är förmodligen så, eftersom att vi får samma acc
            # member.eval(), should be called if we have dropout or batch-norm
            # in our layers, to make sure that self.train = False,
            # just that it doesn't work for now
            self.add_member(member)


