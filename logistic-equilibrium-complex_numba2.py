from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyvista as pv

simulation_length = 5000

equilibrium_resolution = 500
num_simulations = 1000

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
                4 * r * population[i, j, k % size] * (1 - population[i, j, k % size])
            )
            # if not np.isfinite(population[i, j, (k + 1) % size]):
            #    break

    return population


def main():
    print("Hello from imaginary-bifurcations!")

    population[..., 0] = 0.5
    r_re = np.linspace(-0.5, 1, num_simulations)
    r_im = np.linspace(-0.5, 0.5, num_simulations)

    print("Yay")
    paint(population, r_re, r_im)
    print("Finished calculations")

    fig, ax = plt.subplots()

    print(population[:, 450, 0])

    image = ax.matshow(np.abs(population[:, :, 0]))

    def frame(n: int):
        image.set(data=np.abs(population[:, :, n]))
        title = ax.set_title(f"n = {n}")  # update the plot title
        return (image, title)  # return the Artists that have changed

    ani = FuncAnimation(
        fig, frame, range(1, equilibrium_resolution), interval=100, blit=True
    )
    ani.save("slice2.gif", fps=30, dpi=100)

    pl = pv.Plotter()
    pl.show_axes()
    pl.add_points(population)

    # pl.show()


if __name__ == "__main__":
    main()
