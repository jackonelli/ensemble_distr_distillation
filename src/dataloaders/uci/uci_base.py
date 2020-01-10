"""UCI dataset"""
from abc import abstractmethod
import logging
from pathlib import Path
import numpy as np
import torch.utils.data


class UCIData(torch.utils.data.Dataset):
    """UCI base class"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = self._validate_file_path(file_path)
        self._log = logging.getLogger(self.__class__.__name__)
        self.data = None
        self.num_samples = None
        self.dim = None

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        else:
            self._log.warning("Requesting dataset size on unsplit UCI data")
            return self.load_full_data().shape[0]

    def __getitem__(self, index):
        sample = self.data[index, :]
        input_ = sample[:-1]
        target = sample[-1]
        return (input_, target)

    @staticmethod
    def _validate_file_path(file_path):
        """Validate path"""
        file_path = Path(file_path)
        file_path = file_path.expanduser()
        if not file_path.exists():
            raise ValueError(
                "Dataset file: '{}' does not exist".format(file_path))
        return file_path

    @abstractmethod
    def load_full_data(self):
        """Load UCI data into np array"""
        pass

    def add_datasubset(self, data):
        """Add subset of data to a loader

        Args:
            data (np.array(float32)): Shape (num_samples, dim)
        """

        self.data = data
        self.num_samples, self.dim = self.data.shape

    def create_train_val_split(self, training_samples_ratio):
        """Create two separate dataloaders

        Args:
            training_samples_ratio (float): ratio in (0, 1)
            representing the ratio of training samples in the split,
            i.e. training_samples_ratio = 0.9 corresponds to split with
            90% of samples in training set.
        """
        full_data = self.load_full_data()
        np.random.shuffle(full_data)
        total_size = full_data.shape[0]
        training_set_size = int(total_size * training_samples_ratio)

        training_set = UCIData(self.file_path)
        training_data = full_data[:training_set_size, :]
        training_set.add_datasubset(training_data)

        validation_set = UCIData(self.file_path)
        validation_data = full_data[training_set_size:, :]
        validation_set.add_datasubset(validation_data)

        return (training_set, validation_set)
