from numba import njit, prange
import numpy as np
import pyvista as pv

simulation_length = 5000

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

    grid_re, grid_im, grid_pop = np.meshgrid(
        r_re, r_im, np.arange(equilibrium_resolution)
    )

    stacked = np.stack([grid_re, grid_im, np.abs(population)], axis=3)

    coords = np.reshape(stacked, (-1, 3))

    print(coords, coords.shape)

    pl = pv.Plotter()
    pl.show_axes()
    mesh = pv.PolyData(coords)
    mesh.point_data["height"] = coords[:, 2]
    pl.add_mesh(mesh, scalars="height")

    pl.show()


if __name__ == "__main__":
    main()
