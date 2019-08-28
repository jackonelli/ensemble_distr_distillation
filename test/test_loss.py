import unittest
import torch
import torch_testing as tt
import math
from src import loss


class TestDirichletLoss(unittest.TestCase):
    def test_dirichlet_suff_stats_equal(self):
        target_distribution = torch.tensor([[[0.4, 0.6], [0.4, 0.6]]])
        true_suff_stats = torch.tensor([[math.log(0.4), math.log(0.6)]])
        suff_stats = loss._dirichlet_sufficient_statistics(target_distribution)
        tt.assert_almost_equal(suff_stats, true_suff_stats)

    def test_dirichlet_suff_stats_batch(self):
        # B = 3, N = 2, C = 2
        target_distribution = torch.tensor([[[0.4, 0.6], [0.5, 0.5]],
                                            [[0.9, 0.1], [0.8, 0.2]],
                                            [[0.3, 0.7], [0.3, 0.7]]])
        mean_b_1_k_1 = (math.log(0.4) + math.log(0.5)) / 2
        mean_b_1_k_2 = (math.log(0.6) + math.log(0.5)) / 2
        mean_b_2_k_1 = (math.log(0.9) + math.log(0.8)) / 2
        mean_b_2_k_2 = (math.log(0.1) + math.log(0.2)) / 2
        mean_b_3_k_1 = (math.log(0.3) + math.log(0.3)) / 2
        mean_b_3_k_2 = (math.log(0.7) + math.log(0.7)) / 2
        true_suff_stats = torch.tensor([[mean_b_1_k_1, mean_b_1_k_2],
                                        [mean_b_2_k_1, mean_b_2_k_2],
                                        [mean_b_3_k_1, mean_b_3_k_2]])
        suff_stats = loss._dirichlet_sufficient_statistics(target_distribution)
        tt.assert_almost_equal(suff_stats, true_suff_stats)

    def test_dirichlet_nll_uniform(self):
        target_distribution = torch.tensor([[[0.5, 0.5]]])
        alphas = torch.tensor([1, 1], dtype=torch.float)
        dir_nll = loss.dirichlet_neg_log_likelihood(alphas,
                                                    target_distribution)
        self.assertAlmostEqual(dir_nll.item(), 0.0)

    def test_dirichlet_nll_informative(self):
        target_distribution = torch.tensor([[[0.4, 0.6]]])
        alphas = torch.tensor([[3, 1]], dtype=torch.float)
        dir_nll = loss.dirichlet_neg_log_likelihood(alphas,
                                                    target_distribution)
        true_loss = -(math.log(6) - math.log(2) + 2 * math.log(0.4))
        self.assertAlmostEqual(dir_nll.item(), true_loss, places=5)

    def test_dirichlet_nll_batch(self):
        target_distribution = torch.tensor([[[0.4, 0.6], [0.4, 0.6]]])
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
