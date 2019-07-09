import unittest
import torch
import torch_testing as tt
import math
from src import loss


class TestLoss(unittest.TestCase):
    def test_dirichlet_nll(self):
        target_distribution = torch.tensor([0.5, 0.5])
        alphas = torch.tensor([1, 1], dtype=torch.float)
        dir_nll = loss.dirichlet_neg_log_likelihood(alphas,
                                                    target_distribution)
        self.assertAlmostEqual(dir_nll.item(), 0.0)

        target_distribution = torch.tensor([0.4, 0.6])
        alphas = torch.tensor([3, 1], dtype=torch.float)
        dir_nll = loss.dirichlet_neg_log_likelihood(alphas,
                                                    target_distribution)
        true_loss = -math.log(6) + math.log(2) - 2 * math.log(0.4)
        self.assertAlmostEqual(dir_nll.item(), true_loss, places=5)

    def test_dirichlet_nll_batch(self):
        target_distribution = torch.tensor([[0.4, 0.6], [0.4, 0.6]])
        alphas = torch.tensor([[3, 1], [3, 1]], dtype=torch.float)
        dir_nll = loss.dirichlet_neg_log_likelihood(alphas,
                                                    target_distribution)
        true_loss = -math.log(6) + math.log(2) - 2 * math.log(0.4)
        self.assertAlmostEqual(dir_nll.item(), 2 * true_loss, places=5)

    def test_dirichlet_flat_prior(self):
        alphas = torch.tensor([[1, 1], [1, 1]], dtype=torch.float)
        flat_prior = loss.flat_prior(alphas)
        true_loss = 0
        tt.assert_almost_equal(
            torch.ones(alphas.size(0)) * true_loss, flat_prior)


if __name__ == '__main__':
    unittest.main()
