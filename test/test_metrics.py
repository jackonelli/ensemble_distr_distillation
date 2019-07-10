import unittest
from pathlib import Path
import sys
sys.path.append(str(Path.cwd() / "src"))
import math
import torch
import torch_testing as tt
from src import utils
from src import metrics


class TestMetricsFunctions(unittest.TestCase):
    def test_entropy(self):
        predictions = torch.tensor([0.3, 0.7])
        entropy = metrics.entropy(None, predictions)
        self.assertAlmostEqual(entropy.item(), 0.61086430205)

    def test_entropy_batch(self):
        predictions = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
        entropy = metrics.entropy(None, predictions)
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


class TestMetricsClass(unittest.TestCase):
    def test_init(self):
        metric = metrics.Metric("aba", math.log)
        self.assertAlmostEqual(metric.running_value, 0.0)

    def test_generate_metrics(self):
        key = "accuracy"
        metrics_dict = metrics.MetricsDict()
        metrics_dict.add_by_keys(key)
        self.assertEqual(metrics_dict[key].label, key)
        self.assertEqual(metrics_dict[key].function, metrics.accuracy)

    def test_add_measurements(self):
        key = "accuracy"
        metrics_dict = metrics.MetricsDict()
        metrics_dict.add_by_keys(key)
        true_label = torch.tensor([1, 0, 2, 0])
        predictions = torch.tensor([[0.05, 0.09, 0.05], [0.1, 0.8, 0.1],
                                    [0.1, 0.2, 0.7], [0.25, 0.5, 0.25]])
        metric_inst = metrics_dict.pop(key)
        metric_inst.add_sample(true_label, predictions)
        self.assertAlmostEqual(metric_inst.running_value, 0.5)
        self.assertEqual(metric_inst.counter, 4)


if __name__ == '__main__':
    unittest.main()
