import time
from typing import Tuple, Callable

import numpy as np
from numpy import ndarray


def exact(n: int, hit_fn: Callable[[],
          Tuple[bool, float, float]]) -> Tuple[ndarray, float, float]:
    samples = []
    hits = 0
    tested = 0

    start_time = time.time()

    while hits < n:
        tested += 1
        accepted, x, y = hit_fn()

        if accepted:
            samples.append(x)
            hits += 1

    time_duration = time.time() - start_time
    success_rate = (hits / tested) * 100

    return np.array(samples), success_rate, time_duration


def with_limit(n: int, hit_fn: Callable[[],
               Tuple[bool, float, float]]) -> Tuple[ndarray, float, float]:
    samples = []
    hits = 0

    start_time = time.time()

    for _ in range(n):
        accepted, x, y = hit_fn()

        if accepted:
            samples.append(x)
            hits += 1

    time_duration = time.time() - start_time
    success_rate = (hits / n) * 100

    return np.array(samples), success_rate, time_duration
