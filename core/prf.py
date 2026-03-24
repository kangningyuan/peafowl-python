import hmac
import hashlib
from typing import Union


class PRF:
    def __init__(self, key: Union[bytes, int]):
        if isinstance(key, int):
            self.key = key.to_bytes(16, 'big')
        else:
            self.key = key

    def eval(self, input_bytes: bytes) -> int:
        h = hmac.new(self.key, input_bytes, hashlib.sha256)
        return int.from_bytes(h.digest()[:16], 'big')

    def eval_mod(self, input_bytes: bytes, modulus: int) -> int:
        return self.eval(input_bytes) % modulus
