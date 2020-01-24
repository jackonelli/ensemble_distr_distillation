"""UCI dataset"""
from abc import abstractmethod
import logging
from pathlib import Path
import numpy as np
import torch.utils.data
import src.transforms as transforms


class UCIData(torch.utils.data.Dataset):
    """UCI base class"""
    def __init__(self, file_path):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.file_path = file_path
        self._validate_file_path()
        self.x_data = None
        self.y_data = None
        self.num_samples = None
        self.input_dim = None
        self.output_dim = 1
        self.transforms = list()

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        else:
            self._log.warning("Requesting dataset size on unsplit UCI data")
            return self.load_full_data().shape[0]

    def __getitem__(self, index):
        input_ = self.x_data[index, :]
        target = self.y_data[index]
        return (input_, target)

    def _validate_file_path(self):
        """Validate path"""
        file_path = Path(self.file_path)
        self.file_path = file_path.expanduser()
        if not self.file_path.exists():
            self._log.error(
                "File path '{}' to dataset '{}' does not exist:".format(
                    self.file_path, self))

    @abstractmethod
    def load_full_data(self):
        """Load UCI data into np array"""
        pass

    def add_datasubset(self, data):
        """Add subset of data to a loader

        Args:
            data (np.array(float32)): Shape (num_samples, dim)
        """

        self.x_data = data[:, :-1]
        self.y_data = data[:, -1]
        self.num_samples, self.input_dim = self.x_data.shape

    def create_train_val_split(self, training_samples_ratio, normalize=True):
        """Create two separate dataloaders

        Args:
            training_samples_ratio (float): ratio in (0, 1)
            representing the ratio of training samples in the split,
            i.e. training_samples_ratio = 0.9 corresponds to split with
            90% of samples in training set.
        """
        full_data = self.load_full_data()
        total_size = full_data.shape[0]
        training_set_size = int(total_size * training_samples_ratio)

        training_set = UCIData(self.file_path)
        training_data = full_data[:training_set_size, :]
        training_set.add_datasubset(training_data)
        x_mean =
        training_set.add_transform(transforms.Normalize())

        validation_set = UCIData(self.file_path)
        validation_data = full_data[training_set_size:, :]
        validation_set.add_datasubset(validation_data)

        return (training_set, validation_set)

    def add_transform(self, transform):
        if not issubclass(Transform, type(transform)):
            self._log.error("Transform not a subclass of transform.Transform")
        else:
            self.transforms.append(transform)

    def transform(self):
        for transform in self.transforms:
            self.x_data = transform.transform(self.x_data)

            if transform.transform_target:
                self.y_data = transform.transform(self.y_data)

    def calculate_statistics(self, normalize_y=True):
        x_mean = np.mean(self.x_data, 0)
        x_std = np.std(self.x_data, 0)
        x_std[self.x_std == 0] = 1
        y_mean = np.mean(self.y_data)
        y_std = np.std(self.y_data)

        return x_mean, x_std, y_mean, y_std
