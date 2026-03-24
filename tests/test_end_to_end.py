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
        intersection_size = 50
        for i in range(3):
            all_ids = [f"sample_{j}" for j in range(100)]
            if i == 0:
                ids = all_ids
            else:
                common = [f"sample_{j}" for j in range(intersection_size)]
                unique = [f"party{i}_unique_{j}" for j in range(50)]
                ids = common + unique
            features = np.random.randn(len(ids), 20).astype(np.float32)
            dp = DataProvider(f"P{i}", self.config, ids, features)
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
        min_size = min(features.shape[0] for features in aligned.values())
        self.assertEqual(min_size, 50)


if __name__ == '__main__':
    unittest.main()
