import taichi as ti
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "newcent",
})


ti.init(arch=ti.gpu)

n = 50
population = ti.field(dtype=ti.f32, shape=(n,))


@ti.kernel
def paint(r: float):
    for _ in range(1):
        for i in range(n):
            population[i + 1] = 4 * r * population[i] * (1 - population[i])


def main():
    print("Hello from imaginary-bifurcations!")

    population[0] = 0.5
    paint(0.7) # single equilibrium value
    #paint(0.755) # two eq. values

    fig, ax = plt.subplots()

    plt.axhline(0.6428572, c="gray", ls = ":")
    ax.plot(population.to_numpy(), ".-")
    ax.set_xlabel(r"\bf{Generation}")
    ax.set_ylabel(r"\bf{Population Proportion}")
    
    #plt.yticks(np.linspace(0.5, 0.7, 5))
    #ax.set_xscale("log")

    plt.show()


if __name__ == "__main__":
    main()
