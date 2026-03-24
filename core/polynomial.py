from typing import List, Set, Tuple
from .secret_sharing import reconstruct


class Polynomial:
    def __init__(self, coefficients: List[int], modulus: int):
        self.coeffs = coefficients
        self.modulus = modulus
        self.degree = len(coefficients) - 1

    def evaluate(self, x: int) -> int:
        result = 0
        for c in reversed(self.coeffs):
            result = (result * x + c) % self.modulus
        return result

    def evaluate_batch(self, points: List[int]) -> List[int]:
        return [self.evaluate(x) for x in points]


def lagrange_interpolation(points: List[int], values: List[int], modulus: int) -> List[int]:
    if len(points) != len(values):
        raise ValueError("Points and values must have same length")
    if len(points) == 0:
        return []

    n = len(points)
    result = [0] * n

    for i in range(n):
        numerator = 1
        denominator = 1
        for j in range(n):
            if i != j:
                numerator = (numerator * (points[j])) % modulus
                denominator = (denominator * ((points[i] - points[j]) % modulus)) % modulus
        diff = (points[i] * denominator) % modulus
        inv_diff = pow(diff, -1, modulus)
        coeff = (values[i] * numerator * inv_diff) % modulus
        result[i] = coeff

    return result


def evaluate_poly_at_points(coeffs: List[int], points: List[int], modulus: int) -> List[int]:
    results = []
    for x in points:
        result = 0
        for c in reversed(coeffs):
            result = (result * x + c) % modulus
        results.append(result)
    return results


def construct_zero_poly(A: Set[int], n: int, t: int, modulus: int) -> Polynomial:
    all_points = set(range(1, n + 1))
    zero_points = list(all_points - A)
    zero_values = [0] * len(zero_points)
    coeffs = lagrange_interpolation(zero_points, zero_values, modulus)
    return Polynomial(coeffs, modulus)
