from typing import Tuple

import numpy as np

import acceptance_rejection
import pdf


def hit_and_miss() -> Tuple[bool, float, float]:
    x = np.random.uniform(-4, 4)
    y = np.random.uniform(0, 1)

    if y <= pdf.get(x):
        return True, x, y

    return False, x, y


if __name__ == '__main__':
    N = 10_000

    results, success_percents, duration = acceptance_rejection.with_limit(N, hit_and_miss)
    print(f'Hit-and-Miss Method: {len(results)} samples generated in {duration:.2f}'
          f' seconds with a success rate of {success_percents:.2f} %')

    results, success_percents, duration = acceptance_rejection.exact(N, hit_and_miss)
    print(f'Hit-and-Miss Method: {len(results)} samples generated in {duration:.2f}'
          f' seconds with a success rate of {success_percents:.2f} %')
