import unittest
import secrets
import numpy as np
from protocol.przs import PRZS
from protocol.peafowl import PEAFOWL
from protocol.peafowl_offline import PEAFOWLOffline
from party.data_provider import DataProvider
from party.server import Server


class TestPRZS(unittest.TestCase):
    def test_przs_setup(self):
        num_parties = 3
        threshold = 1
        modulus = 2**64
        przs, keys = PRZS.setup(num_parties, threshold, modulus)
        self.assertIsNotNone(przs)
        self.assertIsNotNone(keys)

    def test_przs_generate_share(self):
        num_parties = 3
        threshold = 1
        modulus = 2**64
        przs, keys = PRZS.setup(num_parties, threshold, modulus)
        share = przs.generate_share(0, 1)
        self.assertIsInstance(share, int)


class TestPEAFOWL(unittest.TestCase):
    def setUp(self):
        self.config = {
            'num_parties': 3,
            'num_samples': 100,
            'num_features': 50,
            'secret_modulus': 2**64,
            'precision_bits': 16,
            'shprg_d': 8,
            'shprg_q': 2**128,
            'shprg_p': 2**64,
            'prf_key_bytes': 16,
        }

    def test_peafowl_init(self):
        peafowl = PEAFOWL(self.config)
        self.assertEqual(peafowl.num_parties, 3)
        self.assertEqual(peafowl.modulus, 2**64)

    def test_encrypt_ids(self):
        peafowl = PEAFOWL(self.config)
        ids = ["sample_0", "sample_1", "sample_2"]
        features = np.random.randn(3, 10).astype(np.float32)
        dp = DataProvider("P0", self.config, ids, features)
        encrypted = peafowl.encrypt_ids(dp)
        self.assertIsNotNone(encrypted)

    def test_align_ids_offline(self):
        peafowl = PEAFOWL(self.config)
        server = Server("S", self.config)
        import pickle
        common_ids = {1, 2, 3}
        encrypted_ids = {
            "P0": pickle.dumps({1, 2, 3, 4}),
            "P1": pickle.dumps({1, 2, 3, 5}),
            "P2": pickle.dumps({1, 2, 3, 6}),
        }
        permutations, intersection_size = peafowl.align_ids_offline(server, encrypted_ids)
        self.assertEqual(intersection_size, 3)


class TestPEAFOWLOffline(unittest.TestCase):
    def setUp(self):
        self.config = {
            'num_parties': 3,
            'secret_modulus': 2**64,
            'shprg_d': 8,
            'shprg_q': 2**128,
            'shprg_p': 2**64,
            'shprg_m': 1024,
        }

    def test_precompute_seeds(self):
        offline = PEAFOWLOffline(self.config)
        seeds = offline.precompute_seeds()
        self.assertEqual(len(seeds), 3)
        self.assertEqual(len(seeds[0]), 8)

    def test_generate_permutation_keys(self):
        offline = PEAFOWLOffline(self.config)
        keys = offline.generate_permutation_keys(100)
        self.assertEqual(len(keys), 100)


if __name__ == '__main__':
    unittest.main()
