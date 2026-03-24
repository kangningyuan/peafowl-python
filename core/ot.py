import secrets
from typing import List, Tuple
from .prf import PRF


class OTExtension:
    def __init__(self, security_parameter: int = 128):
        self.security_parameter = security_parameter
        self.kappa = security_parameter // 8

    def extend_ot(self, messages: List[Tuple[bytes, bytes]], receiver_choice_bits: List[int]) -> List[bytes]:
        if len(messages) != len(receiver_choice_bits):
            raise ValueError("Number of messages must match number of choice bits")

        t_keys = [secrets.token_bytes(self.kappa) for _ in range(len(messages))]
        results = []

        for i, (msg_pair, choice) in enumerate(zip(messages, receiver_choice_bits)):
            prf = PRF(t_keys[i])
            if choice == 0:
                results.append(msg_pair[0] + prf.eval(msg_pair[1]))
            else:
                results.append(msg_pair[1] + prf.eval(msg_pair[0]))

        return results

    def simulate_ot(self, num_ot: int) -> Tuple[List[Tuple[bytes, bytes]], List[int]]:
        messages = [
            (secrets.token_bytes(self.kappa), secrets.token_bytes(self.kappa))
            for _ in range(num_ot)
        ]
        receiver_bits = [secrets.randbelow(2) for _ in range(num_ot)]
        return messages, receiver_bits
