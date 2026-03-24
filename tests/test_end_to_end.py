import unittest
import numpy as np
from protocol.peafowl import PEAFOWL
from party.data_provider import DataProvider
from party.server import Server


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.config = {
            'num_parties': 3,
            'num_samples': 100,
            'num_features': 20,
            'secret_modulus': 2**64,
            'precision_bits': 16,
            'shprg_d': 8,
            'shprg_q': 2**128,
            'shprg_p': 2**64,
            'prf_key_bytes': 16,
        }
        np.random.seed(42)
        self.data_providers = []
        common_ids = [f"sample_{j}" for j in range(50)]
        for i in range(3):
            ids = common_ids + [f"party{i}_unique_{j}" for j in range(50)]
            features = np.random.randn(len(ids), 20).astype(np.float32)
            dp = DataProvider(f"P{i}", self.config, ids, features)
            dp.prf_key = b'0' * 16
            self.data_providers.append(dp)
        self.server = Server("S", self.config)

    def test_end_to_end_alignment(self):
        peafowl = PEAFOWL(self.config)
        aligned = peafowl.run_protocol(self.data_providers, self.server)
        self.assertEqual(len(aligned), 3)
        for party_id, features in aligned.items():
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(features.shape[1], 20)

    def test_intersection_size(self):
        peafowl = PEAFOWL(self.config)
        aligned = peafowl.run_protocol(self.data_providers, self.server)
        for dp in self.data_providers:
            aligned_ids = dp.get_aligned_ids()
            self.assertTrue(len(aligned_ids) > 0)


if __name__ == '__main__':
    unittest.main()
