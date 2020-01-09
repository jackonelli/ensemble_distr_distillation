import unittest
import numpy as np
import src.dataloaders.uci.wine as uci_wine


class TestWineData(unittest.TestCase):
    def setUp(self):
        base = uci_wine.WineData(
            "~/doktor/datasets/UCI/wine/winequality-red.csv")
        self.training_set, self.validation_set = base.create_train_val_split(
            0.7)

    def test_num_samples(self):
        self.assertEqual(len(self.training_set), 1119)
        self.assertEqual(len(self.validation_set), 1599 - 1119)

    def test_dim(self):
        input_, target = self.training_set[0]
        self.assertEqual(input_.shape[0], 11)
        self.assertIsInstance(target, np.float32)


if __name__ == '__main__':
    unittest.main()
