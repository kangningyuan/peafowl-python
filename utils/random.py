import secrets


def secure_randbelow(upper_bound: int) -> int:
    return secrets.randbelow(upper_bound)


def secure_randint(low: int, high: int) -> int:
    return secrets.randbelow(high - low + 1) + low


def secure_random_bytes(length: int) -> bytes:
    return secrets.token_bytes(length)
