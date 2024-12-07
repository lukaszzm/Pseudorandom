from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

import box_muller


def generate_sums(distribution_func: Callable[[int], np.ndarray],
                  size: int, sums: int) -> [float]:
    sums = [np.sum(distribution_func(size)) for _ in range(sums)]
    return np.array(sums)


def normal_distribution(size: int) -> [float]:
    return box_muller.generate(size // 2)[:size]


def exponential_distribution(size: int) -> [float]:
    return np.random.exponential(scale=1.0, size=size)


def uniform_distribution(size: int) -> [float]:
    return np.random.uniform(low=0.0, high=1.0, size=size)


def get_differences(sorted_values: [float]) -> [float]:
    return np.diff(sorted_values)


def print_sums_histogram(norm_sums: [float], exp_sums: [float],
                         unif_sums: [float], size: int) -> None:
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.hist(norm_sums, bins=50, density=True, alpha=0.6, color='skyblue')
    plt.title(f'Sums of {size} Samples from Normal Distribution')
    plt.xlabel('Sum')
    plt.ylabel('Density')

    plt.subplot(3, 1, 2)
    plt.hist(exp_sums, bins=50, density=True, alpha=0.6, color='lightgreen')
    plt.title(f'Sums of {size} Samples from Exponential Distribution')
    plt.xlabel('Sum')
    plt.ylabel('Density')

    plt.subplot(3, 1, 3)
    plt.hist(unif_sums, bins=50, density=True, alpha=0.6, color='lightcoral')
    plt.title(f'Sums of {size} Samples from Uniform Distribution')
    plt.xlabel('Sum')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()


def print_diff_histogram(differences: [float], title: str) -> None:
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.hist(differences, bins=100, density=True, alpha=0.6, color='skyblue')
    plt.title(title)
    plt.xlabel('Difference')
    plt.ylabel('Density')

    plt.subplot(2, 1, 2)
    plt.hist(differences, bins=100, density=True, alpha=0.6, color='skyblue')
    plt.yscale('log')
    plt.title(f'{title} (Log Scale)')
    plt.xlabel('Difference')
    plt.ylabel('Density (Log Scale)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # A)
    sample_size = 10
    sums_count = 10_000

    normal_sums = generate_sums(normal_distribution, sample_size, sums_count)
    exponential_sums = generate_sums(exponential_distribution, sample_size, sums_count)
    uniform_sums = generate_sums(uniform_distribution, sample_size, sums_count)

    print_sums_histogram(normal_sums, exponential_sums, uniform_sums, sample_size)

    sample_size = 100

    normal_sums = generate_sums(normal_distribution, sample_size, sums_count)
    exponential_sums = generate_sums(exponential_distribution, sample_size,
                                     sums_count)
    uniform_sums = generate_sums(uniform_distribution, sample_size, sums_count)

    print_sums_histogram(normal_sums, exponential_sums, uniform_sums, sample_size)

    # Central Limit Theorem

    # B)

    sample_size = 10_000_000

    sorted_normal = np.sort(normal_distribution(sample_size))
    sorted_exponential = np.sort(exponential_distribution(sample_size))
    sorted_uniform = np.sort(uniform_distribution(sample_size))

    diff_normal = get_differences(sorted_normal)
    diff_exponential = get_differences(sorted_exponential)
    diff_uniform = get_differences(sorted_uniform)

    print_diff_histogram(diff_normal, 'Differences of Sorted Normal Distribution')
    print_diff_histogram(diff_exponential, 'Differences of Sorted Exponential '
                                           'Distribution')
    print_diff_histogram(diff_uniform, 'Differences of Sorted Uniform Distribution')
