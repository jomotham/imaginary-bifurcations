import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu)

plt.rcParams.update({
    "mathtext.default": "it",
    "mathtext.fontset": "cm",
    "font.family": "Ebrima",
})


simulation_length = 2000
num_simulations = 50000
arr_type = ti.types.ndarray(ndim=1)

population = ti.field(dtype=ti.f32, shape=(num_simulations, simulation_length))

@ti.kernel
def paint(r_values: arr_type):
    for j in r_values:
        r = r_values[j]
        for i in range(simulation_length):
            population[j, i + 1] = r * population[j, i] * (1 - population[j, i])


def full():
    print("Hello from imaginary-bifurcations!")

    for i in range(num_simulations):
        population[i, 0] = 0.5
    r = np.linspace(0.8, 4, num_simulations, dtype=np.float32)
    # r = 1 - np.logspace(0, 0.8, num_simulations, dtype=np.float32)
    paint(r_values=r)
    print("Finished calculations")

    fig, ax = plt.subplots()

    pop = population.to_numpy()
    ax.plot(r, pop[:, -50:], ".", ms=0.03, c='black')
    ax.set_xlim(right=4.05)
    ax.set_ylim(0,1)

    ax.set_xlabel("$r$ value")
    ax.set_ylabel("Equilibrium value(s)")
    ax.set_title("Logisitc Map Bifurcation Diagram")

    fig.set_size_inches(10,6)
    plt.savefig("biffy.png", dpi=300)
    plt.show()
    

def zoomed():
    print("Hello from imaginary-bifurcations!")

    for i in range(num_simulations):
        population[i, 0] = 0.5
    r = np.linspace(3.2, 3.9, num_simulations, dtype=np.float32)
    paint(r_values=r)
    print("Finished calculations")

    fig, ax = plt.subplots()

    pop = population.to_numpy()
    ax.plot(r, pop[:, -50:], ".", ms=0.03, c='black')

    ax.set_xlim(3.2, 3.9)

    ax.set_xlabel("$r$ value")
    ax.set_ylabel("Equilibrium value(s)")
    ax.set_title("Logisitc Map Bifurcation Diagram")

    fig.set_size_inches(10,6)
    plt.savefig("biffy_zoomed.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    full()
    #zoomed()
