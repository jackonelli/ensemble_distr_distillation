"""UCI dataset"""
from abc import abstractmethod
import logging
from pathlib import Path
import numpy as np
import torch.utils.data as torch_data


class UCIData():
    """UCI base class"""
    def __init__(self, file_path, seed=0):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self.file_path = self._validate_file_path(file_path)
        self.data = None
        self.input_dim = None
        self.output_dim = 1
        self.seed = seed
        self.load_full_data()

    def _validate_file_path(self, file_path):
        """Validate path"""
        file_path = Path(file_path)
        file_path = file_path.expanduser()
        if not file_path.exists():
            self._log.error("Dataset does not exist")
        return file_path

    @abstractmethod
    def load_full_data(self):
        """Load UCI data into np array"""
        pass


class _UCIDataset(torch_data.Dataset):
    """Internal representation of a subset of UCI data"""
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.num_samples, self.input_dim = self.x_data.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index, :]


def uci_dataloader(x_data, y_data, batch_size):
    """Generate a dataloader"""
    dataset = _UCIDataset(x_data, y_data)
    return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
