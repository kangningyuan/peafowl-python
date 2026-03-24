import secrets
from typing import List


def share(secret: int, n: int, modulus: int) -> List[int]:
    shares = [secrets.randbelow(modulus) for _ in range(n - 1)]
    last = (secret - sum(shares)) % modulus
    shares.append(last)
    return shares


def reconstruct(shares: List[int], modulus: int) -> int:
    return sum(shares) % modulus


def share_vector(vector: List[int], n: int, modulus: int) -> List[List[int]]:
    return [share(v, n, modulus) for v in vector]


def reconstruct_vector(shares: List[List[int]], modulus: int) -> List[int]:
    if not shares:
        return []
    return [reconstruct(shares[k], modulus) for k in range(len(shares))]
