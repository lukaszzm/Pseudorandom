import numpy as np


def generate(shape: [float]) -> [float]:
    u1 = np.random.rand(shape)
    u2 = np.random.rand(shape)

    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    return np.concatenate((z1, z2))
