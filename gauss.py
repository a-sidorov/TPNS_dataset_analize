from math import sqrt, exp, pi
from typing import Callable


def gauss(mean: float, variance: float) -> Callable[[float], float]:
    return lambda x: 1 / sqrt(2 * pi * variance) * exp(-((x - mean) ** 2) / (2 * variance))
