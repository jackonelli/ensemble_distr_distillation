import unittest
import torch
from src import utils
from src import metrics


class TestMetrics(unittest.TestCase):
    def test_entropy(self):
        predictions = torch.tensor([0.3, 0.7])
        entropy = metrics.entropy(predictions)
        self.assertAlmostEqual(entropy.item(), 0.61086430205)

    def test_nll(self):
        true_label = torch.tensor(1)
        num_classes = 2
        true_one_hot = utils.to_one_hot(true_label, num_classes)
        predictions = torch.tensor([0.5, 0.5])
        nll = metrics.nll(true_one_hot, predictions)
        self.assertAlmostEqual(nll.item(), 0.69314718)


if __name__ == '__main__':
    unittest.main()
