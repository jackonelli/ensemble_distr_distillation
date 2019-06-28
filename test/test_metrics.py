import unittest
import torch
import torch_testing as tt
from src import utils
from src import metrics


class TestMetrics(unittest.TestCase):
    def test_entropy(self):
        predictions = torch.tensor([0.3, 0.7])
        entropy = metrics.entropy(predictions)
        self.assertAlmostEqual(entropy.item(), 0.61086430205)

    def test_entropy_batch(self):
        predictions = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
        entropy = metrics.entropy(predictions)
        tt.assert_almost_equal(entropy,
                               torch.tensor([0.61086430205, 0.61086430205]))

    def test_nll(self):
        true_label = torch.tensor(1)
        num_classes = 2
        true_one_hot = utils.to_one_hot(true_label, num_classes)
        predictions = torch.tensor([0.5, 0.5])
        nll = metrics.nll(true_one_hot, predictions)
        self.assertAlmostEqual(nll.item(), 0.69314718)

    def test_nll_batch(self):
        true_label = torch.tensor([1, 1])
        num_classes = 2
        true_one_hot = utils.to_one_hot(true_label, num_classes)
        predictions = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        nll = metrics.nll(true_one_hot, predictions)
        tt.assert_almost_equal(nll, 0.69314718 * torch.ones((2)))


if __name__ == '__main__':
    unittest.main()
