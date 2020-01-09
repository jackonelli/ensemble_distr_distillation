"""UCI Wine dataset

https://archive.ics.uci.edu/ml/datasets/Wine
"""

from pathlib import Path
from src.dataloaders.uci.uci_base import UCIData


class WineData(UCIData):
    """Dataloader for wine data

    Args:
        file_path (str / pathlib.Path)
    """
    def __init__(self, file_path):
        super().__init__(file_path)
