import numpy as np


def get(x: float) -> float:
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)


def get_array(x: [float]) -> [float]:
    return [get(i) for i in x]
