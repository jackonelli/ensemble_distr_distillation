"""Test Inverse-Wishart (in combination with normal)"""
import torch
import unittest
import src.loss as loss
import numpy as np
import math

NUM_DECIMALS = 5


class TestWishartLoss(unittest.TestCase):
    """Test inverse wishart only"""
    def test_wishart_nll_one_dim(self):
        target = torch.tensor([[3.5]], dtype=torch.float)
        psi = torch.tensor([[4.0]], dtype=torch.float)
        nu = torch.tensor([[10]], dtype=torch.float)

        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu), target)
        self.assertAlmostEqual(wish_nll.item(),
                               np.math.lgamma(5) + 1.982816,
                               places=NUM_DECIMALS)

    def test_wishart_nll_two_dim(self):
        target = torch.tensor([[3.5, 2.0]], dtype=torch.float)
        psi = torch.tensor([[4.0, 2.5]], dtype=torch.float)
        nu = torch.tensor([[10]], dtype=torch.float)

        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu), target)
        self.assertAlmostEqual(wish_nll.item(),
                               np.math.lgamma(5) + 3.06673,
                               places=NUM_DECIMALS)

    def test_wishart_nll_two_ensemble_members(self):
        target = torch.unsqueeze(torch.tensor([[3.5, 2.5]], dtype=torch.float),
                                 dim=-1)
        psi = torch.tensor([[4.0]], dtype=torch.float)
        nu = torch.tensor([[10.0]], dtype=torch.float)

        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu), target)
        self.assertAlmostEqual(wish_nll.item(),
                               np.math.lgamma(5) + 1.24737,
                               places=NUM_DECIMALS)

    def test_wishart_nll_batch(self):
        target = torch.tensor([[3.5], [2.5]], dtype=torch.float)
        psi = torch.tensor([[4.0], [2.0]], dtype=torch.float)
        nu = torch.tensor([[10.0], [5.0]], dtype=torch.float)

        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu), target)
        self.assertAlmostEqual(wish_nll.item(),
                               0.5 * np.math.lgamma(5) +
                               0.5 * np.math.lgamma(2.5) + 1.991126,
                               places=NUM_DECIMALS)


class TestNormalInverseWishartLoss(unittest.TestCase):
    """Test normal-inverse wishart"""
    def test_wishart_nll_one_dim(self):
        cov_mat_target = torch.tensor(8, dtype=torch.float).reshape(1, 1)
        mu_target = torch.tensor(0.0, dtype=torch.float).reshape(1, 1, 1)

        mu_0 = torch.tensor(-1.0, dtype=torch.float).reshape(1, 1)
        lambda_ = torch.tensor(2.0, dtype=torch.float).reshape(1, 1)

        gauss_nll = loss.gaussian_neg_log_likelihood((mu_0,
                                                      cov_mat_target / lambda_),
                                                     mu_target)

        self.assertAlmostEqual(gauss_nll.item(),
                               math.log(2) + math.log(2 * math.pi) / 2 + 1 / 8,
                               places=NUM_DECIMALS)

        psi = torch.tensor([[4.0]], dtype=torch.float)
        nu = torch.tensor([[10]], dtype=torch.float)

        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu),
                                                           cov_mat_target)

        self.assertAlmostEqual(wish_nll.item(),
                               np.math.lgamma(5) + 1.982816,
                               places=NUM_DECIMALS)

    def test_wishart_nll_two_dim(self):
        target = torch.tensor([[3.5, 2.0]], dtype=torch.float)
        psi = torch.tensor([[4.0, 2.5]], dtype=torch.float)
        nu = torch.tensor([[10]], dtype=torch.float)

        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu), target)
        self.assertAlmostEqual(wish_nll.item(),
                               np.math.lgamma(5) + 3.06673,
                               places=NUM_DECIMALS)

    def test_wishart_nll_two_ensemble_members(self):
        target = torch.unsqueeze(torch.tensor([[3.5, 2.5]], dtype=torch.float),
                                 dim=-1)
        psi = torch.tensor([[4.0]], dtype=torch.float)
        nu = torch.tensor([[10.0]], dtype=torch.float)

        dim = 1
        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu), target)
        self.assertAlmostEqual(wish_nll.item(),
                               np.math.lgamma(5) + 5 * math.log(2) - 5 * torch.log(4),
                               places=NUM_DECIMALS)

    def test_wishart_nll_batch(self):
        target = torch.tensor([[3.5], [2.5]], dtype=torch.float)
        psi = torch.tensor([[4.0], [2.0]], dtype=torch.float)
        nu = torch.tensor([[10.0], [5.0]], dtype=torch.float)

        wish_nll = loss.inverse_wishart_neg_log_likelihood((psi, nu), target)
        self.assertAlmostEqual(wish_nll.item(),
                               0.5 * np.math.lgamma(5) +
                               0.5 * np.math.lgamma(2.5) + 1.991126,
                               places=NUM_DECIMALS)


if __name__ == '__main__':
    unittest.main()
