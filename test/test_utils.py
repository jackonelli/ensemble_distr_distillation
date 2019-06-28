import unittest
import torch
from src import utils


class TestMetrics(unittest.TestCase):
    def test_to_one_hot(self):
        label = torch.tensor(1)
        num_classes = 2
        one_hot = utils.to_one_hot(label, num_classes)
        self.assertEqual(one_hot.shape[0], num_classes)
        self.assertEqual(one_hot[0], 0)
        self.assertEqual(one_hot[1], 1)


if __name__ == '__main__':
    unittest.main()
