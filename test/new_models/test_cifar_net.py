import unittest
from src.new_models import cifar_net


class TestMetrics(unittest.TestCase):
    def test_init(self):
        cifar_net.EnsembleNet()
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
