import unittest
from math import sqrt
import numpy as np
import numpy.testing as npt
import src.dataloaders.uci.wine as uci_wine
import src.dataloaders.uci.bost as uci_bost
import src.dataloaders.uci.uci_base as uci_base
from copy import deepcopy

NUM_DECIMALS = 5


class TestWineData(unittest.TestCase):
    def setUp(self):
        base = uci_wine.WineData(
            "~/doktor/datasets/UCI/wine/winequality-red.csv")
        self.training_set, self.validation_set = base.create_train_val_split(
            0.7, normalize=False)

    def test_num_samples(self):
        self.assertEqual(len(self.training_set), 1119)
        self.assertEqual(len(self.validation_set), 1599 - 1119)

    def test_dim(self):
        input_, target = self.training_set[0]
        self.assertEqual(input_.shape[0], 11)
        self.assertIsInstance(target, np.float32)


class TestBostonData(unittest.TestCase):
    def setUp(self):
        base = uci_bost.BostonData("~/doktor/datasets/UCI/bost/housing.data")
        self.training_set, self.validation_set = base.create_train_val_split(
            0.7, normalize=False)

    def test_num_samples(self):
        self.assertEqual(len(self.training_set), 354)
        self.assertEqual(len(self.validation_set), 506 - 354)

    def test_dim(self):
        input_, target = self.training_set[0]
        self.assertEqual(input_.shape[0], 13)
        self.assertIsInstance(target, np.float32)

    def test_first_row(self):
        input_, target = self.training_set[0]
        self.assertAlmostEqual(input_[0].item(), 0.00632, places=NUM_DECIMALS)
        self.assertAlmostEqual(target, 24.00, places=NUM_DECIMALS)
        self.assertIsInstance(target, np.float32)


class TestBaseClass(unittest.TestCase):
    def setUp(self):
        base = uci_base.UCIData("dummy")
        dummy_data = np.array([[1, 1, 1, 0], [1, 2, 3, 1], [2, 2, 2, 2]])
        base.add_datasubset(dummy_data)
        self.base = base
        self.dummy_data = dummy_data

    def test_normalize_x(self):
        base = deepcopy(self.base)
        base.normalize()
        npt.assert_array_almost_equal(base.x_mean,
                                      np.array([4 / 3, 5 / 3, 2]),
                                      decimal=NUM_DECIMALS)

        npt.assert_array_almost_equal(
            base.x_std,
            np.array([sqrt(2 / 9), sqrt(2 / 9),
                      sqrt(2 / 3)]),
            decimal=NUM_DECIMALS)

        self.assertAlmostEqual(base.y_mean, 1)
        self.assertAlmostEqual(base.y_std, sqrt(2 / 3))
        npt.assert_array_almost_equal(
            base.x_data, (self.dummy_data[:, :-1] - base.x_mean) / base.x_std)

    def test_normalize_with_y(self):
        base = deepcopy(self.base)
        base.normalize()

        self.assertAlmostEqual(base.y_mean, 1)
        self.assertAlmostEqual(base.y_std, sqrt(2 / 3))
        npt.assert_array_almost_equal(base.y_data,
                                      np.array([-sqrt(3 / 2), 0,
                                                sqrt(3 / 2)]))

    def test_normalize_without_y(self):
        base = deepcopy(self.base)
        base.normalize(normalize_y=False)
        npt.assert_array_almost_equal(base.y_data, np.array([0, 1, 2]))


if __name__ == '__main__':
    unittest.main()
