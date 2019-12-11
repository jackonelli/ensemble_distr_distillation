import torch
import unittest
import math
import src.loss as loss

NUM_DECIMALS = 0


class TestGaussianLoss(unittest.TestCase):
    def test_gaussian_nll_one_dim(self):
        B, N, D = 1, 1, 1
        target = torch.tensor([[1.0]], dtype=torch.float).reshape(B, N, D)
        mean = torch.tensor([[0.5]], dtype=torch.float)
        var = torch.tensor([[10]], dtype=torch.float)

        gauss_nll = loss.gaussian_neg_log_likelihood((mean, var), target)
        self.assertAlmostEqual(gauss_nll.item(),
                               0.5 * math.log(2 * math.pi * 10) + 0.0125,
                               places=NUM_DECIMALS)

    def test_gaussian_nll_two_dim(self):
        pass
        B, N, D = 1, 1, 2
        target = torch.tensor([[1.0, 2.0]], dtype=torch.float).reshape(B, N, D)
        mean = torch.tensor([[0.5, 1.0]], dtype=torch.float)
        var = torch.tensor([[2.0, 5.0]], dtype=torch.float)

        gauss_nll = loss.gaussian_neg_log_likelihood((mean, var), target)
        self.assertAlmostEqual(gauss_nll.item(),
                               math.log(2 * math.pi) + 0.5 * math.log(10) +
                               0.1625,
                               places=NUM_DECIMALS)

    def test_gaussian_nll_two_ensemble_members(self):
        B, N, D = 1, 2, 1
        target = torch.unsqueeze(torch.tensor([[1.0, 0.75]],
                                              dtype=torch.float),
                                 dim=-1).reshape(B, N, D)
        mean = torch.tensor([[0.5]], dtype=torch.float)
        var = torch.unsqueeze(torch.tensor([10.0], dtype=torch.float), dim=-1)

        gauss_nll = loss.gaussian_neg_log_likelihood((mean, var), target)
        self.assertAlmostEqual(gauss_nll.item(),
                               0.25 * math.log(4 * math.pi**2 * 10**2) +
                               (0.015625 / 2),
                               places=NUM_DECIMALS)

    def test_gaussian_nll_batch(self):
        pass
        B, N, D = 2, 1, 1
        target = torch.tensor([[1.0], [0.75]],
                              dtype=torch.float).reshape(B, N, D)
        mean = torch.tensor([[0.5], [0.25]], dtype=torch.float)
        var = torch.tensor([[10.0], [5.0]], dtype=torch.float)

        gauss_nll = loss.gaussian_neg_log_likelihood((mean, var), target)
        self.assertAlmostEqual(gauss_nll.item(),
                               0.25 * math.log(4 * math.pi**2 * 10 * 5) +
                               0.01875,
                               places=NUM_DECIMALS)

    # def test_gaussian_nll_unopt(self):
    #     target = torch.tensor([[[1.0], [0.75]]], dtype=torch.float)
    #     mean = torch.tensor([[0.5], [0.25]], dtype=torch.float)
    #     var = torch.tensor([[10.0], [5.0]], dtype=torch.float)

    #     gauss_nll = loss.gaussian_neg_log_likelihood_unopt((mean, var), target)
    #     self.assertAlmostEqual(gauss_nll.item(),
    #                            0.25 * math.log(4 * math.pi**2 * 10 * 5) +
    #                            0.01875,
    #                            places=5)


if __name__ == '__main__':
    unittest.main()
