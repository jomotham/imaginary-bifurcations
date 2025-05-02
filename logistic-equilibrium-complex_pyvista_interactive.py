from numba import njit, prange
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from dataclasses import dataclass

import time


@njit(parallel=True)
def paint(population, r_real, r_imag, simulation_length):
    for idx in prange(r_real.size * r_imag.size):
        # i = idx // r_real.size
        # j = idx % r_real.size
        i = idx % r_real.size
        j = idx // r_real.size
        r = r_real[i] + 1j * r_imag[j]
        for k in range(simulation_length):
            size = population.shape[2]
            x_n = population[i, j, k % size]

            # z = r * (1 / 2 - x_n)
            # c = r / 2 * (1 - r / 2)

            # value = z**2 + c

            value = r * x_n * (1 - x_n)
            population[i, j, (k + 1) % size] = value

    return population


@dataclass
class Parameters:
    simulation_length: int
    equilibrium_resolution: int
    num_simulations: int
    limits: tuple[complex, complex]

    @property
    def r_real(self):
        return np.linspace(*np.real(self.limits), self.num_simulations)

    @property
    def r_imag(self):
        return np.linspace(*np.imag(self.limits), self.num_simulations)


def simulate(params: Parameters, population: np.ndarray):
    desired_shape = (
        params.num_simulations,
        params.num_simulations,
        params.equilibrium_resolution,
    )

    if (
        population is None
        or population.shape != desired_shape
        or population.dtype != np.complex64
    ):
        population = np.zeros(
            dtype=np.complex64,
            shape=desired_shape,
        )

    population[..., 0] = 0.5

    print("Yay")
    start_time = time.perf_counter()
    paint(population, params.r_real, params.r_imag, params.simulation_length)
    end_time = time.perf_counter()
    print(f"Finished calculations in {(end_time - start_time) * 1000} ms")

    return population


def create_mesh(params: Parameters, population: np.ndarray):
    grid_re, grid_im, grid_pop = np.meshgrid(
        params.r_real, params.r_imag, np.arange(params.equilibrium_resolution)
    )

    stacked = np.stack([grid_re, grid_im, np.real(population)], axis=3)

    coords = np.reshape(stacked, (-1, 3))

    phases = np.reshape(np.angle(population), (-1, 1))

    highlight = grid_pop
    highlight[:, :, 2:] = 0
    highlight = np.reshape(highlight, (-1, 1))

    print("Making mesh")
    mesh = pv.PolyData(coords)
    mesh.point_data["height"] = coords[:, 2]
    mesh.point_data["angles"] = phases
    mesh.point_data["highlight"] = highlight

    return mesh


if __name__ == "__main__":
    print("Hello from imaginary-bifurcations!")

    params = Parameters(
        simulation_length=5000,
        equilibrium_resolution=6,
        num_simulations=800,
        limits=((-2 - 2j), (4 + 2j)),
    )

    print(params)

    population = simulate(params, None)
    mesh = create_mesh(params, population)

    pl = BackgroundPlotter()
    pl.show_axes()

    print("Inserting mesh")
    pl.add_mesh(mesh, scalars="angles", name="mymesh")
    pl.show_bounds(
        location="outer",
        xtitle="real",
        ytitle="imaginary",
        ztitle="equilibrium value",
    )
    pl.show()

    # print("Creating gif")
    # viewup = [0.5, 0.5, 1]
    # path = pl.generate_orbital_path(
    #     factor=2.0, shift=mesh.length, viewup=viewup, n_points=36
    # )
    # pl.open_gif("orbit.gif")
    # pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
    # pl.close()

    pl.app.exec_()
