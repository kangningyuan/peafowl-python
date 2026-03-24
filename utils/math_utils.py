def mod_inverse(a: int, modulus: int) -> int:
    if a < 0:
        a = a % modulus
    g, x, _ = extended_gcd(a, modulus)
    if g != 1:
        raise ValueError("Modular inverse does not exist")
    return x % modulus


def extended_gcd(a: int, b: int):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def gaussian_elimination(matrix, modulus: int = None):
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    result = [row[:] for row in matrix]
    lead = 0
    for r in range(rows):
        if lead >= cols:
            break
        i = r
        while result[i][lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if lead >= cols:
                    break
        if lead >= cols:
            break
        result[i], result[r] = result[r], result[i]
        lv = result[r][lead]
        if modulus:
            lv_inv = mod_inverse(lv, modulus)
            for j in range(cols):
                result[r][j] = (result[r][j] * lv_inv) % modulus
        else:
            for j in range(cols):
                result[r][j] = result[r][j] / lv
        for i in range(rows):
            if i != r:
                lv = result[i][lead]
                if modulus:
                    for j in range(cols):
                        result[i][j] = (result[i][j] - lv * result[r][j]) % modulus
                else:
                    for j in range(cols):
                        result[i][j] = result[i][j] - lv * result[r][j]
        lead += 1
    return result
