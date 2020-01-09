"""UCI Wine dataset

https://archive.ics.uci.edu/ml/datasets/Wine
"""

from pathlib import Path
from src.dataloaders.uci.uci_base import UCIData


class WineData(UCIData):
    """Dataloader for wine data"""
    def __init__(self):
        super().__init__(
            file_path=Path("~/doktor/datasets/UCI/wine/winequality-red.csv"))
