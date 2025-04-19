from numba import jit
import numpy as np
import matplotlib.pyplot as plt


n = 50
population = np.zeros(dtype=np.float32, shape=(n,))


@jit
def paint(population, r: float):
    for i in range(n - 1):
        population[i + 1] = 4 * r * population[i] * (1 - population[i])
    return population


def main():
    print("Hello from imaginary-bifurcations!")

    population[0] = 0.1
    paint(population, 0.7)

    fig, ax = plt.subplots()

    ax.plot(population, ".")

    plt.show()


if __name__ == "__main__":
    main()
