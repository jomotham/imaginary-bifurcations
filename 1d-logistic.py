import taichi as ti
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "mathtext.default": "it",
    "mathtext.fontset": "cm",
    "font.family": "Ebrima",
})


ti.init(arch=ti.gpu)

n = 100
population = ti.field(dtype=ti.f32, shape=(n,))

r = 0.87

@ti.kernel
def paint(r: float):
    for _ in range(1):
        for i in range(n):
            population[i + 1] = 4 * r * population[i] * (1 - population[i])


def main():
    print("Hello from imaginary-bifurcations!")

    population[0] = 0.5
    paint(r) # single equilibrium value
    #paint(0.755) # two eq. values

    fig, ax = plt.subplots()

    # plt.axhline(0.713, c="gray", ls = ":")
    # plt.axhline(0.618, c="gray", ls = ":")
    ax.plot(population.to_numpy(), ".-")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Proportion")
    ax.set_title(f"Logistic Map with $r={r}$")
    
    #plt.yticks(np.linspace(0.5, 0.7, 5))
    #ax.set_xscale("log")



    plt.axhline(0.87, c="gray", ls = ":")
    plt.axhline(0.833, c="gray", ls = ":")
    plt.axhline(0.486, c="gray", ls = ":")
    plt.axhline(0.394, c="gray", ls = ":")

    plt.savefig("logi4eq.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
