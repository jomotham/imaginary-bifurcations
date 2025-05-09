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

#1eq: 2.80
#2eq: 3.02
#4eq: 3.48

r = 3.48

@ti.kernel
def paint(r: float):
    for _ in range(1):
        for i in range(n):
            population[i + 1] = r * population[i] * (1 - population[i])


def main():
    print("Hello from imaginary-bifurcations!")

    population[0] = 0.5
    paint(r)

    fig, ax = plt.subplots()
    
    
    ax.plot(population.to_numpy(), ".-",label='Population Proportion')
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population Proportion")
    ax.set_title(f"Logistic Map with $r={r}$")
    ax.legend()

    #plt.axhline(0.6428, c="gray", ls = ":")

    # plt.axhline(0.713, c="gray", ls = ":")
    # plt.axhline(0.618, c="gray", ls = ":")
    

    plt.axhline(0.87, c="gray", ls = ":")
    plt.axhline(0.833, c="gray", ls = ":")
    plt.axhline(0.486, c="gray", ls = ":")
    plt.axhline(0.394, c="gray", ls = ":")
    ax.set_ylim(top=0.93)

    plt.savefig("logi4eq.png", dpi=300)
    plt.show()
    


if __name__ == "__main__":
    main()
    
