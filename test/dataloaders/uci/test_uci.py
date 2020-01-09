import unittest
import src.dataloaders.uci.wine as uci_wine


class TestWineData(unittest.TestCase):
    def setUp(self):
        self.data = uci_wine.WineData()

    def test_num_samples(self):
        self.assertEqual(len(self.data), 1599)

    def test_dim(self):
        input_, target = self.data[0]
        self.assertEqual(len(input_), 11)
        self.assertEqual(len(target), 1)


if __name__ == '__main__':
    unittest.main()
