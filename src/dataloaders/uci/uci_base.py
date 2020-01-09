"""UCI dataset"""
import csv
import logging
import numpy as np
import torch.utils.data


class UCIData(torch.utils.data.Dataset):
    """UCI base class"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path.expanduser()
        self._log = logging.getLogger(self.__class__.__name__)

    def __len__(self):
        with self.file_path.open(newline="") as csv_file:
            csv_list = list(csv.reader(csv_file, delimiter=";", quotechar="|"))
        return len(csv_list) - 1

    def __getitem__(self, index):
        sample = None
        with self.file_path.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=";", quotechar="|")
            next(csv_reader)  # Skip header
            for count, row in enumerate(csv_reader):
                if count == index:
                    sample = row
                    break
        inputs = sample[:-1]
        targets = sample[-1]

        return (np.array(inputs, dtype=np.float32),
                np.array([targets], dtype=np.float32))
