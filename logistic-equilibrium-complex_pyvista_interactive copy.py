from numba import njit, prange
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from dataclasses import dataclass

import time


@njit(parallel=True)
def paint(population, simulation_length):
    for idx in prange(population.shape[0]):
        sim_arr = population[idx, 2, :]
        r = population[idx, 0, 0] + 1j * population[idx, 1, 0]
        for k in range(simulation_length):
            size = len(sim_arr)
            x_n = sim_arr[k % size]
            value = 4 * r * x_n * (1 - x_n)
            sim_arr[(k + 1) % size] = value

            if not np.isfinite(value):
                sim_arr[:] = np.nan
                break

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


def create_buffer(params: Parameters):
    grid_re, grid_im, grid_pop = np.meshgrid(
        params.r_real,
        params.r_imag,
        np.zeros(shape=(params.equilibrium_resolution,), dtype=np.complex64),
    )

    population = np.stack([grid_re, grid_im, grid_pop], axis=2)
    population = np.reshape(
        population, shape=(-1, 3, params.equilibrium_resolution), copy=True
    )

    print(population.shape)

    return population


def create_mesh(params: Parameters, population: np.ndarray):
    # print(population, np.swapaxes(population, 1, 2).shape)
    coords = np.reshape(np.swapaxes(population, 1, 2), (-1, 3))
    # check that we haven't made a copy
    assert coords.base is not None

    # real_coords = coords.view(np.float64)
    # strided_view = real_coords[:, 0:-1:2]
    # print(real_coords.shape, strided_view.shape)
    print(coords.shape)

    mesh = pv.PolyData(np.real(coords))
    mesh.point_data["height"] = np.abs(coords[:, 2])
    mesh.point_data["angles"] = np.angle(coords[:, 2])

    return mesh


def run_simulations(params: Parameters, plotter: pv.Plotter, id: str):
    population = create_buffer(params)

    for decimation in reversed(range(1, 50)):
        decimated_population = population[::decimation]
        decimated_population[..., 2, :] = 0.5

        print("Yay")
        start_time = time.perf_counter()
        paint(decimated_population, params.simulation_length)
        end_time = time.perf_counter()
        print(f"Finished calculations in {(end_time - start_time) * 1000} ms")

        mesh = create_mesh(params, decimated_population)

        print("Inserting mesh")
        plotter.add_mesh(mesh, scalars="angles", name=f"mymesh{decimation}{id}")
        plotter.render()
        plotter.app.processEvents()


def main():
    print("Hello from imaginary-bifurcations!")

    pl = BackgroundPlotter()
    pl.show_axes()
    pl.show_bounds(
        location="outer",
        xtitle="real",
        ytitle="imaginary",
        ztitle="equilibrium value",
    )
    pl.show()
    # print(params)

    params = Parameters(
        simulation_length=5000,
        equilibrium_resolution=6,
        num_simulations=300,
        limits=(-0.5 - 0.5j, 1 + 0.5j),
    )

    run_simulations(params, pl, "base")

    params.limits = (0.5 - 0.5j, 1 + 0.5j)

    run_simulations(params, pl, "run1")

    # print("Creating gif")
    # viewup = [0.5, 0.5, 1]
    # path = pl.generate_orbital_path(
    #     factor=2.0, shift=mesh.length, viewup=viewup, n_points=36
    # )
    # pl.open_gif("orbit.gif")
    # pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
    # pl.close()
    print("Yielding to eventloop")
    pl.app.exec_()


if __name__ == "__main__":
    main()
