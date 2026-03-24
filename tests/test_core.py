import unittest
import secrets
from core.secret_sharing import share, reconstruct, share_vector, reconstruct_vector
from core.prf import PRF
from core.shprg import SHPRG
from core.ot import OTExtension
from core.permute_share import PermuteShare, ShareTranslator
from core.polynomial import Polynomial, lagrange_interpolation, evaluate_poly_at_points


class TestSecretSharing(unittest.TestCase):
    def test_share_and_reconstruct(self):
        modulus = 2**64
        secret = secrets.randbelow(modulus)
        n = 3
        shares = share(secret, n, modulus)
        self.assertEqual(len(shares), n)
        reconstructed = reconstruct(shares, modulus)
        self.assertEqual(reconstructed, secret)

    def test_share_vector(self):
        modulus = 2**64
        vector = [1, 2, 3, 4, 5]
        n = 3
        shares = share_vector(vector, n, modulus)
        self.assertEqual(len(shares), len(vector))
        self.assertEqual(len(shares[0]), n)
        reconstructed = reconstruct_vector(shares, modulus)
        self.assertEqual(reconstructed, vector)


class TestPRF(unittest.TestCase):
    def test_prf_eval(self):
        key = secrets.token_bytes(16)
        prf = PRF(key)
        input_bytes = b"test_input"
        output1 = prf.eval(input_bytes)
        output2 = prf.eval(input_bytes)
        self.assertEqual(output1, output2)

    def test_prf_different_keys(self):
        key1 = secrets.token_bytes(16)
        key2 = secrets.token_bytes(16)
        prf1 = PRF(key1)
        prf2 = PRF(key2)
        input_bytes = b"test_input"
        self.assertNotEqual(prf1.eval(input_bytes), prf2.eval(input_bytes))

    def test_prf_mod(self):
        key = secrets.token_bytes(16)
        prf = PRF(key)
        modulus = 2**64
        output = prf.eval_mod(b"test", modulus)
        self.assertLess(output, modulus)


class TestSHPRG(unittest.TestCase):
    def test_shprg_generate(self):
        d, m, q, p = 8, 1024, 2**128, 2**64
        shprg = SHPRG(d, m, q, p)
        seed = [secrets.randbelow(q) for _ in range(d)]
        result = shprg.generate(seed)
        self.assertEqual(len(result), m)
        self.assertTrue(all(0 <= x < p for x in result))

    def test_shprg_same_seed(self):
        d, m, q, p = 8, 128, 2**128, 2**64
        shprg = SHPRG(d, m, q, p)
        seed = [secrets.randbelow(q) for _ in range(d)]
        result1 = shprg.generate(seed)
        result2 = shprg.generate(seed)
        self.assertEqual(result1, result2)

    def test_shprg_combine_seeds(self):
        d, m, q, p = 8, 128, 2**128, 2**64
        shprg = SHPRG(d, m, q, p)
        seed1 = [secrets.randbelow(q) for _ in range(d)]
        seed2 = [secrets.randbelow(q) for _ in range(d)]
        combined = SHPRG.combine_seeds(seed1, seed2, q)
        self.assertEqual(len(combined), d)


class TestOTExtension(unittest.TestCase):
    def test_simulate_ot(self):
        ot = OTExtension(security_parameter=128)
        messages, bits = ot.simulate_ot(10)
        self.assertEqual(len(messages), 10)
        self.assertEqual(len(bits), 10)

    def test_extend_ot(self):
        ot = OTExtension(security_parameter=128)
        num_ot = 5
        messages = [
            (secrets.token_bytes(16), secrets.token_bytes(16))
            for _ in range(num_ot)
        ]
        choice_bits = [secrets.randbelow(2) for _ in range(num_ot)]
        results = ot.extend_ot(messages, choice_bits)
        self.assertEqual(len(results), num_ot)


class TestPermuteShare(unittest.TestCase):
    def test_share_translator(self):
        modulus = 2**64
        translator = ShareTranslator(modulus)
        pi = [1, 0, 2]
        delta = translator.translate(pi, 'S')
        self.assertEqual(len(delta), len(pi))
        result = translator.translate(pi, 'P')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestPolynomial(unittest.TestCase):
    def test_lagrange_interpolation(self):
        points = [1, 2, 3]
        values = [6, 11, 18]
        modulus = 2**64
        coeffs = lagrange_interpolation(points, values, modulus)
        self.assertEqual(len(coeffs), 3)
        results = evaluate_poly_at_points(coeffs, points, modulus)
        self.assertEqual(results, values)

    def test_polynomial_evaluate(self):
        coeffs = [1, 2, 3]
        modulus = 2**64
        poly = Polynomial(coeffs, modulus)
        result = poly.evaluate(2)
        expected = 1 + 2*2 + 3*4
        self.assertEqual(result, expected % modulus)


if __name__ == '__main__':
    unittest.main()
