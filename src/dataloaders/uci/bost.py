"""UCI Boston housing dataset

http://lib.stat.cmu.edu/datasets/boston
"""

import numpy as np
from src.dataloaders.uci.uci_base import UCIData


class BostonData(UCIData):
    """Dataloader for Boston housing data

    Args:
        file_path (str / pathlib.Path)
    """
    def __init__(self, file_path):
        super().__init__(file_path)

    def load_full_data(self):
        """Load csv data into np array"""
        return np.genfromtxt(self.file_path,
                             dtype=np.float32,
                             delimiter=" ",
                             skip_header=0)
