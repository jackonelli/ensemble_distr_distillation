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

    def test_accuracy(self):
        true_label = torch.tensor(0)
        predictions = torch.tensor([0.9, 0.1])
        acc = metrics.accuracy(true_label, predictions)
        self.assertAlmostEqual(acc, 1)

    def test_accuracy_batch(self):
        true_label = torch.tensor([1, 0, 2, 0])
        predictions = torch.tensor([[0.05, 0.09, 0.05], [0.1, 0.8, 0.1],
                                    [0.1, 0.2, 0.7], [0.25, 0.5, 0.25]])
        acc = metrics.accuracy(true_label, predictions)
        self.assertAlmostEqual(acc, 0.5)

    def test_squared_error(self):
        targets = torch.tensor([1, 2, 1.5])
        predictions = torch.tensor([[0.9, 2.1, 1.7, 0.0, 0.0, 0.0]])
        squared_error = metrics.squared_error(targets, predictions)
        self.assertAlmostEqual(squared_error, 0.02)


if __name__ == '__main__':
    unittest.main()
