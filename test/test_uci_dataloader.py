import unittest
from src.dataloaders.uci.wine import WineData

NUM_DECIMALS = 5


class TestWineLoader(unittest.TestCase):
    def test_wine_loader(self):
        wine_data = WineData()
        first_row_x = wine_data[0][0]
        first_row_y = wine_data[0][1]
        self.assertAlmostEqual(first_row_x[0], 7.4, places=NUM_DECIMALS)
        self.assertAlmostEqual(first_row_y[0], 5, places=NUM_DECIMALS)
