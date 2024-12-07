import numpy as np

import acceptance_rejection
import pdf


def lorentzian_pdf(x: float, c: float) -> float:
    return c / (np.pi * (x ** 2 + 1))


def lorentzian_hit_and_miss() -> tuple[bool, float, float]:
    c = 1.52

    x = np.random.uniform(-10, 10)
    y = np.random.uniform(0, lorentzian_pdf(0, c))

    if y <= pdf.get(x):
        return True, x, y

    return False, x, y


if __name__ == '__main__':
    N = 10_000

    results, success_percents, duration = (
        acceptance_rejection.with_limit(N, lorentzian_hit_and_miss))

    print(
        f'Hybrid Hit-and-Miss Method: {len(results)} samples generated in {duration:.2f}'
        f' seconds with a success rate of {success_percents:.2f} %')

    results, success_percents, duration = (
        acceptance_rejection.exact(N, lorentzian_hit_and_miss))

    print(
        f'Hybrid Hit-and-Miss Method: {len(results)} samples generated in {duration:.2f}'
        f' seconds with a success rate of {success_percents:.2f} %')
