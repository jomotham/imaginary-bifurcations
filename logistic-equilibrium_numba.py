from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt

# ti.init(arch=ti.gpu)

simulation_length = 200_000

equilibrium_resolution = 50
num_simulations = 50000

population = np.zeros(dtype=np.float32, shape=(num_simulations, equilibrium_resolution))


@njit(parallel=True)
def paint(population, r_values):
    for j in prange(r_values.shape[0]):
        r = r_values[j]
        for i in range(simulation_length):
            size = equilibrium_resolution
            population[j, (i + 1) % size] = (
                4 * r * population[j, i % size] * (1 - population[j, i % size])
            )

    return population


def main():
    print("Hello from imaginary-bifurcations!")

    population[:, 0] = 0.5
    r = np.linspace(0.2, 1, num_simulations, dtype=np.float32)
    # r = 1 - np.logspace(0, 0.8, num_simulations, dtype=np.float32)
    paint(population, r_values=r)
    print("Finished calculations")

    fig, ax = plt.subplots()

    ax.plot(population[:, -50:], ".", ms=2)

    plt.show()


if __name__ == "__main__":
    main()
