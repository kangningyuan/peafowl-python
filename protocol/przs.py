import secrets
from typing import Dict, List, Set, Tuple, Any
from core.prf import PRF
from core.polynomial import Polynomial, lagrange_interpolation


class PRZS:
    def __init__(self, keys: Dict[Tuple[int, ...], List[bytes]], n: int, t: int, modulus: int):
        self.keys = keys
        self.n = n
        self.t = t
        self.modulus = modulus
        self.precomputed_polys = self._precompute_polynomials()

    def _precompute_polynomials(self) -> Dict[Tuple[int, ...], List[List[int]]]:
        poly_cache = {}
        for A_tuple, key_list in self.keys.items():
            A = set(A_tuple)
            all_points = set(range(1, self.n + 1))
            zero_points = list(all_points - A)
            zero_values = [0] * len(zero_points)
            coeffs = lagrange_interpolation(zero_points, zero_values, self.modulus)
            poly_cache[A_tuple] = coeffs
        return poly_cache

    def generate_share(self, party_id: int, point: int) -> int:
        share = 0
        for A_tuple, key_list in self.keys.items():
            if party_id in A_tuple:
                coeffs = self.precomputed_polys.get(A_tuple, [])
                prf = PRF(key_list[0])
                r = prf.eval_mod(point.to_bytes(16, 'big'), self.modulus)
                if coeffs:
                    poly = Polynomial(coeffs, self.modulus)
                    f_val = poly.evaluate(party_id)
                else:
                    f_val = 0
                share = (share + r * f_val) % self.modulus
        return share

    @staticmethod
    def setup(num_parties: int, threshold: int, modulus: int) -> Tuple['PRZS', Dict[Tuple[int, ...], List[bytes]]]:
        keys = {}
        for combo in range(num_parties):
            for t_size in range(threshold + 1):
                subset = tuple([i for i in range(num_parties) if i != combo][:t_size])
                if len(subset) == t_size:
                    key = secrets.token_bytes(16)
                    if subset not in keys:
                        keys[subset] = []
                    keys[subset].append(key)
        przs = PRZS(keys, num_parties, threshold, modulus)
        return przs, keys
