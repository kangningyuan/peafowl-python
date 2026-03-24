import secrets
from typing import List, Tuple, Any
from .secret_sharing import share, reconstruct


class ShareTranslator:
    def __init__(self, modulus: int):
        self.modulus = modulus

    def translate(self, pi: List[int], party_id: str) -> Any:
        if party_id == 'S':
            delta = [secrets.randbelow(self.modulus) for _ in range(len(pi))]
            return delta
        else:
            a = [secrets.randbelow(self.modulus) for _ in range(len(pi))]
            b = [secrets.randbelow(self.modulus) for _ in range(len(pi))]
            return (a, b)


class PermuteShare:
    def __init__(self, modulus: int):
        self.modulus = modulus
        self.translator = ShareTranslator(modulus)

    def permute_and_share(self, pi: List[int], x: List[int], party_id: str, delta: Any = None) -> List[int]:
        if party_id == 'S':
            if delta is None:
                delta = [secrets.randbelow(self.modulus) for _ in range(len(pi))]
            permuted_x = [x[pi[i]] if pi[i] < len(x) else x[0] for i in range(len(pi))]
            result = [(permuted_x[i] + delta[i]) % self.modulus for i in range(len(pi))]
            return result
        else:
            a, b = delta if delta else (secrets.randbelow(self.modulus), secrets.randbelow(self.modulus))
            m = [(x[i] - a[i]) % self.modulus for i in range(len(x))]
            result = b[:len(x)]
            return result

    def permute_share_server(self, pi: List[int], x: List[int], delta: List[int]) -> List[int]:
        permuted_x = [x[pi[i]] if pi[i] < len(x) else x[0] for i in range(len(pi))]
        return [(permuted_x[i] + delta[i]) % self.modulus for i in range(len(pi))]

    def permute_share_client(self, x: List[int], a: List[int], b: List[int]) -> List[int]:
        m = [(x[i] - a[i]) % self.modulus for i in range(len(x))]
        return b[:len(x)]
