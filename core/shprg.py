import secrets
from typing import List
from .prf import PRF


class SHPRG:
    def __init__(self, d: int, m: int, q: int, p: int):
        self.d = d
        self.m = m
        self.q = q
        self.p = p
        self.A = [[secrets.randbelow(q) for _ in range(m)] for _ in range(d)]
        self.shift = q // p

    def generate(self, seed: List[int]) -> List[int]:
        intermediate = [0] * self.m
        for col in range(self.m):
            s = 0
            for row in range(self.d):
                s = (s + self.A[row][col] * seed[row]) % self.q
            intermediate[col] = s
        result = [(x // self.shift) % self.p for x in intermediate]
        return result

    def generate_batch(self, seeds: List[List[int]]) -> List[List[int]]:
        return [self.generate(seed) for seed in seeds]

    @staticmethod
    def combine_seeds(seed1: List[int], seed2: List[int], modulus: int) -> List[int]:
        return [(s1 + s2) % modulus for s1, s2 in zip(seed1, seed2)]
