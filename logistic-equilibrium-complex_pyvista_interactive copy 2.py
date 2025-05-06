from numba import njit, prange
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from dataclasses import dataclass

import time


@njit(parallel=True)
def paint(population, simulation_length):
    for idx in prange(population.shape[0]):
        x_0 = population[idx, 0, 0] + 1j * population[idx, 1, 0]
        r = 1.5
        sim_arr = population[idx, 2, :]
        size = len(sim_arr)
        sim_arr[:] = x_0
        for k in range(simulation_length):
            x_n = sim_arr[k % size]
            value = r * x_n * (1 - x_n)
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


def run_simulations(params: Parameters, plotter: pv.Plotter, id: str, num_steps: int):
    population = create_buffer(params)

    for decimation in reversed(range(1, num_steps)):
        decimated_population = population[:: decimation**2]
        decimated_population[..., 2, :] = 0.5

        print("Calculating...")
        start_time = time.perf_counter()
        paint(decimated_population, params.simulation_length)
        end_time = time.perf_counter()
        print(f"Finished calculations in {(end_time - start_time) * 1000} ms")

        mesh = create_mesh(params, decimated_population)

        print("Inserting mesh")
        plotter.add_mesh(
            mesh, scalars="height", name=f"mymesh{decimation}{id}", cmap="gist_earth"
        )

        plotter.render()
        plotter.write_frame()
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
    pl.enable_parallel_projection()
    pl.show()
    pl.open_gif("points.gif")
    # print(params)

    params = Parameters(
        simulation_length=5000,
        equilibrium_resolution=6,
        num_simulations=800,
        limits=((-2 - 2j), (4 + 2j)),
    )

    run_simulations(params, pl, "base", num_steps=2)

    # params.limits = ((3 - 0.5j), (4 + 0.5j))
    # params.equilibrium_resolution = 50
    # run_simulations(params, pl, "run1", num_steps=10)

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
    print("Saving gif")

    pl.close()


if __name__ == "__main__":
    main()
