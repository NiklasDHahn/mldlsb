from typing import Iterable
from numpy.typing import ArrayLike
import math

import numpy as np
import matplotlib.pyplot as plt

from mldlsb.actfun import sigmoid, hyperbolic_tangent, relu, leaky_relu, elu


def tickrange(min: float, max: float) -> ArrayLike:
    rng = abs(max - min)
    step = math.ceil(rng / 2) if rng > 1 else round(rng / 2, 1)
    print(step)

    return np.arange(min, max + 0.2, step=step)  # Add 0.1 to include max


def visualize(seq: Iterable[float], function: str, **kwargs) -> None:
    # All implemented functions
    functions = {
        "sigmoid": sigmoid,
        "hyperbolic tangent": hyperbolic_tangent,
        "relu": relu,
        "leaky relu": leaky_relu,
        "elu": elu,
    }

    # Sanity
    function = function.lower()
    if function not in functions:
        raise NotImplementedError

    # function_values = np.array(list(map(functions[function], seq, kwargs)))
    function_values = np.array([functions[function](s=s, **kwargs) for s in seq])

    # seq_min = np.min(seq)
    # seq_max = np.max(seq)
    # f_min = np.min(function_values)
    # f_max = np.max(function_values)

    # Position of horizontal and vertical lines
    # x_vert = 0
    # y_hori = 0.5

    # Set figure size
    size = kwargs["size"] if "size" in kwargs else (9, 3)

    # Set graph color
    color = kwargs["color"] if "color" in kwargs else "#00FA9A"

    title = f"{function.capitalize()} Function"
    x_label = "Input Range"
    y_label = "f(x)"
    # x_ticks = tickrange(seq_min, seq_max)
    # y_ticks = tickrange(f_min, f_max)

    # Plot
    plt.figure(figsize=size)
    # plt.axvline(x=x_vert, color="black", linestyle="--", alpha=0.3)
    # plt.axhline(y=y_hori, color="black", linestyle="--", alpha=0.3)
    plt.plot(seq, function_values, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    # plt.xticks([seq_min, 0, seq_max+0.1])
    # plt.yticks([round(f_min, 1), 0.5, round(f_max, 1)])
    plt.show()
