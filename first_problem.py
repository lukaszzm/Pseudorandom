import numpy as np
import matplotlib.pyplot as plt

import box_muller
import pdf


def print_histogram_with_pdf(samples: [float]) -> None:
    plt.figure(figsize = (10, 6))
    counts, bins, _ = plt.hist(samples, bins = 50, density = True, alpha = 0.6,
                               color = 'skyblue', label = 'Histogram')

    x_axis = np.linspace(min(bins), max(bins), 1000)
    y_axis = pdf.get_array(x_axis)
    plt.plot(x_axis, y_axis, color = 'red', linewidth = 2, label = 'Analytical PDF')

    plt.title('Histogram of Generated Numbers and Analytical PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    count = 10_000

    generated_samples = box_muller.generate(count // 2)
    print_histogram_with_pdf(generated_samples)
