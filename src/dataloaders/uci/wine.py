"""UCI Wine dataset

https://archive.ics.uci.edu/ml/datasets/Wine
"""

import numpy as np
from src.dataloaders.uci.uci_base import UCIData


class WineData(UCIData):
    """Dataloader for wine data

    Args:
        file_path (str / pathlib.Path)
    """
    def __init__(self, file_path):
        super().__init__(file_path)

    def load_full_data(self):
        """Load csv data into np array"""
        return np.genfromtxt(self.file_path,
                             dtype=np.float32,
                             delimiter=";",
                             skip_header=1)
