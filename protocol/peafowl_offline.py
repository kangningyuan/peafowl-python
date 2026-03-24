import secrets
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from core.shprg import SHPRG
from core.permute_share import PermuteShare


class PEAFOWLOffline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_parties = config.get('num_parties', 3)
        self.modulus = config.get('secret_modulus', 2**64)
        self.shprg_d = config.get('shprg_d', 8)
        self.shprg_q = config.get('shprg_q', 2**128)
        self.shprg_p = config.get('shprg_p', 2**64)
        self.shprg_m = config.get('shprg_m', 1024)

    def precompute_seeds(self) -> List[List[int]]:
        seeds = []
        for _ in range(self.num_parties):
            seed = [secrets.randbelow(self.shprg_q) for _ in range(self.shprg_d)]
            seeds.append(seed)
        return seeds

    def generate_shprg_matrix(self) -> Tuple[List[List[int]], SHPRG]:
        shprg = SHPRG(self.shprg_d, self.shprg_m, self.shprg_q, self.shprg_p)
        return shprg.A, shprg

    def compute_offline_shares(
        self,
        seeds: List[List[int]],
        party_id: int
    ) -> List[int]:
        shprg = SHPRG(self.shprg_d, self.shprg_m, self.shprg_q, self.shprg_p)
        return shprg.generate(seeds[party_id])

    def generate_permutation_keys(self, num_samples: int) -> List[int]:
        return [secrets.randbelow(self.modulus) for _ in range(num_samples)]

    def offline_permute_share(
        self,
        x: List[int],
        pi: List[int],
        party_id: str,
        delta: Optional[List[int]] = None
    ) -> List[int]:
        permuter = PermuteShare(self.modulus)
        return permuter.permute_and_share(pi, x, party_id, delta)
