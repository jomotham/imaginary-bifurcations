import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu)

simulation_length = 20000
num_simulations = 50000

population = ti.field(dtype=ti.math.vec2, shape=(num_simulations, simulation_length))

arr_type = ti.types.ndarray(ndim=1)

@ti.kernel
def paint(r_values: arr_type):
    for j in r_values:
        r = r_values[j]
        for i in range(simulation_length):
            population[j, i + 1] = 4 * r * population[j, i] * (1 - population[j, i])


def main():
    print("Hello from imaginary-bifurcations!")

    for i in range(num_simulations):
        population[i, 0] = ti.math.vec2(0.5, 0)
    r = np.linspace(0.2, 1, num_simulations, dtype=np.float32)
    # r = 1 - np.logspace(0, 0.8, num_simulations, dtype=np.float32)
    paint(r_values=r)
    print("Finished calculations")

    fig, ax = plt.subplots()

    pop = population.to_numpy()
    ax.plot(r, pop[:, -50:], ".", ms=0.1)

    ax.set_xlabel("$r$ value")
    ax.set_ylabel("Equilibrium value(s)")

    plt.show()


if __name__ == "__main__":
    main()
