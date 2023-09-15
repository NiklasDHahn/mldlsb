import math


def sigmoid(s: float, **kwargs) -> float:
    return 1 / (1 + math.exp(-s))


def hyperbolic_tangent(s: float, **kwargs) -> float:
    a = kwargs["a"] if "a" in kwargs else 2
    return (math.exp(a * s) - 1) / (math.exp(a * s) + 1)


def relu(s: float, **kwargs) -> float:
    return max(0, s)


def leaky_relu(s: float, **kwargs) -> float:
    a = kwargs["a"] if "a" in kwargs else 0.1
    return max(0.1 * s, s)


def elu(s: float, **kwargs) -> float:
    a = kwargs["a"] if "a" in kwargs else 1.0
    if s > 0:
        return s
    return a * (math.exp(s) - 1)
