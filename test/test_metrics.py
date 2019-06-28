import unittest
import torch
from src import utils
from src import metrics


class TestMetrics(unittest.TestCase):
    def test_nll(self):
        true_label = torch.tensor(1)
        num_classes = 2
        true_one_hot = utils.to_one_hot(true_label, num_classes)
        predictions = torch.tensor([0.5, 0.5])
        nll = metrics.nll(true_one_hot, predictions)
        self.assertAlmostEquals(nll.item(), 0.69314718)


if __name__ == '__main__':
    unittest.main()
