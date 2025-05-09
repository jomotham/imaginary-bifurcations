from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    "mathtext.default": "it",
    "mathtext.fontset": "cm",
    "font.family": "Ebrima",
})

simulation_length = 10_000

equilibrium_resolution = 50
num_simulations = 500

population = np.zeros(
    dtype=np.complex64, shape=(num_simulations, num_simulations, equilibrium_resolution)
)


@njit(parallel=True)
def paint(population, r_real, r_imag):
    for idx in prange(r_real.size * r_imag.size):
        i = idx % r_real.size
        j = idx // r_real.size
        r = r_real[i] + 1j * r_imag[j]
        for k in range(simulation_length):
            size = equilibrium_resolution
            population[i, j, (k + 1) % size] = (
                r * population[i, j, k % size] * (1 - population[i, j, k % size])
            )

    return population


def main():
    print("Hello from imaginary-bifurcations!")

    population[..., 0] = 0.5
    r_re = np.linspace(0.8, 4, num_simulations)
    r_im = np.linspace(-1.2, 1.2, num_simulations)

    print("Yay")
    paint(population, r_re, r_im)
    print("Finished calculations")

    fig, ax = plt.subplots()

    lines = ax.plot(
        r_re, np.abs(population[:, num_simulations // 2, :]), ".", ms=0.5, c="black"
    )
    ax.set_xlabel(r"$\mathrm{Re}(r)$")
    ax.set_ylabel("Equilibrium value")

    def frame(n: int):
        print(f"{round(n/num_simulations * 100, 1)}%")
        for i, line in enumerate(lines):
            line.set_ydata(np.abs(population[:, n, i]))
        ax.autoscale(True)
        title = ax.set_title(
            f"$\\mathrm{{Im}}(r) = {'{:.4f}'.format(r_im[n])}$"
        )  # update the plot title
        return (line, title)  # return the Artists that have changed

    ani = FuncAnimation(fig, frame, range(1, num_simulations), interval=100, blit=True)
    ani.save("slice_through_imaginary_axis.gif", fps=30, dpi=300)


if __name__ == "__main__":
    main()
