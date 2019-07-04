import unittest
import torch
import torch_testing as tt
from src import loss


class TestMetrics(unittest.TestCase):
    def test_dirichlet_nll(self):
        target_distribution = torch.tensor([0.5, 0.5])
        alphas = torch.tensor([1, 1], dtype=torch.float)
        dir_nll = loss.dirichlet_neg_log_likelihood(alphas,
                                                    target_distribution)
        self.assertAlmostEqual(dir_nll.item(), 0.0)

    # def test_entropy_batch(self):
    #     predictions = torch.tensor([[0.3, 0.7], [0.7, 0.3]])
    #     entropy = metrics.entropy(predictions)
    #     tt.assert_almost_equal(entropy,
    #                            torch.tensor([0.61086430205, 0.61086430205]))


if __name__ == '__main__':
    unittest.main()
