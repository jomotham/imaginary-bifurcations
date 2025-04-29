from numba import njit, prange
import numpy as np
import pyvista as pv

simulation_length = 5000

equilibrium_resolution = 5
num_simulations = 800

population = np.zeros(
    dtype=np.complex64, shape=(num_simulations, num_simulations, equilibrium_resolution)
)


@njit(parallel=True)
def paint(population, r_real, r_imag):
    for idx in prange(r_real.size * r_imag.size):
        i = idx % r_real.size
        j = idx // r_real.size
        r = 4 * (r_real[i] + 1j * r_imag[j])
        for k in range(simulation_length):
            size = equilibrium_resolution
            value = r * population[i, j, k % size] * (1 - population[i, j, k % size])
            population[i, j, (k + 1) % size] = value

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
    phases = np.reshape(np.angle(population), (-1, 1))

    highlight = grid_pop
    highlight[:, :, 2:] = 0
    highlight = np.reshape(highlight, (-1, 1))

    print(coords.shape)

    pl = pv.Plotter()
    pl.show_axes()
    print("Making mesh")
    mesh = pv.PolyData(coords)
    mesh.point_data["height"] = coords[:, 2]
    mesh.point_data["angles"] = phases
    mesh.point_data["highlight"] = highlight
    print("Inserting mesh")
    pl.add_mesh(mesh, scalars="angles")
    # pl.show_bounds()
    pl.show(auto_close=False)

    print("Creating gif")
    viewup = [0.5, 0.5, 1]
    path = pl.generate_orbital_path(
        factor=2.0, shift=mesh.length, viewup=viewup, n_points=36
    )
    pl.open_gif("orbit.gif")
    pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
    pl.close()


if __name__ == "__main__":
    main()
