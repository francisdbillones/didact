from __future__ import annotations

import math


def add_backward(a: float, b: float, grad: float) -> tuple[float, float]:
    return grad, grad


def mul_backward(a: float, b: float, grad: float) -> tuple[float, float]:
    return grad * b, grad * a


def pow_backward(a: float, b: float, grad: float) -> tuple[float, float]:
    return grad * b * a ** (b - 1), grad * (a ** b) * math.log(a)
