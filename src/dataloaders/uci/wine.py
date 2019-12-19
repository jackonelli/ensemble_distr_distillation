"""UCI Wine dataset

https://archive.ics.uci.edu/ml/datasets/Wine
"""

import logging
import csv
from pathlib import Path
import numpy as np
from src.dataloaders.uci.uci_base import UCIData


class WineData(UCIData):
    """Dataloader for wine data"""
    def __init__(self):
        super().__init__(
            file_path=Path("~/doktor/datasets/UCI/wine/winequality-red.csv"))
        self._log = logging.getLogger(self.__class__.__name__)
        self.n_samples = None

    def __len__(self):
        length = None
        if self.n_samples is not None:
            length = self.n_samples
        else:
            self._log.warning("Requesting sample size without initialisation")
            length = 0
        return length

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
