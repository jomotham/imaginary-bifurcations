from numba import njit, prange
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from dataclasses import dataclass

import time


@njit(parallel=True)
def paint(population, simulation_length):
    for idx in prange(population.shape[0]):
        # logistic growth rate
        r = population[idx, 0, 0] + 1j * population[idx, 1, 0]

        # our scratch space
        sim_arr = population[idx, 2, :]

        size = len(sim_arr)
        for k in range(simulation_length):
            # we may want to simulate for more iterations than we want to save at the end,
            # so use the scratch space as a circular buffer
            x_n = sim_arr[k % size]

            # quadratic map
            c = 0.5 * 4 * (1 - 0.5 * r)
            value = x_n**2 + c
            # value = r * x_n * (1 - x_n)

            sim_arr[(k + 1) % size] = value

            if not np.isfinite(value):
                # it's blown up so we don't need to spend more effort
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
    """Create a list of coordinates sampling the imaginary plane in a grid

    but make it deep in the last dimension as scratch space to run the simulation
    """

    grid_re, grid_im, grid_pop = np.meshgrid(
        params.r_real,
        params.r_imag,
        np.zeros(shape=(params.equilibrium_resolution,), dtype=np.complex64),
    )

    population = np.stack([grid_re, grid_im, grid_pop], axis=2)
    population = np.reshape(
        population, shape=(-1, 3, params.equilibrium_resolution), copy=True
    )

    print(f"Allocated buffer {population.shape}")

    return population


def create_mesh(params: Parameters, population: np.ndarray):
    # flatten the simulation scratch space axis out so we have one long list of coordinates
    coords = np.reshape(np.swapaxes(population, 1, 2), (-1, 3))

    print(f"Calculated {coords.shape} points")

    # strip the nans before giving it to pyvista
    coords = coords[~np.isnan(coords[:, 2])]

    print(f"Created cloud of {coords.shape} points")

    # try to limit outliers that mess up our colormap
    out = np.clip(np.real(coords), -10, 10)

    mesh = pv.PolyData(out)

    # populate "scalars" to use for the colormap
    mesh.point_data["height"] = np.abs(coords[:, 2])
    mesh.point_data["angles"] = np.angle(coords[:, 2])

    return mesh


def run_simulations(params: Parameters, plotter: pv.Plotter, id: str, num_steps: int):
    population = create_buffer(params)

    for decimation in reversed(range(1, num_steps + 1)):
        # create a strided view of the list of coordinates
        decimated_population = population[:: decimation**2]
        decimated_population[..., 2, :] = 0.1

        print("Calculating...")
        start_time = time.perf_counter()
        # run the simulation
        paint(decimated_population, params.simulation_length)
        end_time = time.perf_counter()
        print(f"Finished calculations in {(end_time - start_time) * 1000} ms")

        mesh = create_mesh(params, decimated_population)

        print("Inserting mesh")
        plotter.add_mesh(
            mesh,
            scalars="height",
            name=f"mymesh{decimation}{id}",  # pyvista will deduplicate meshes with the same name
            cmap="gist_earth",
            opacity=0.2,
        )

        plotter.render()
        # plotter.write_frame() # write to the gif
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
    pl.enable_fly_to_right_click()

    def bounds_callback(box):
        print(
            box,
            box.bounds,
        )
        bounds = (
            box.bounds.x_min + 1j * box.bounds.y_min,
            box.bounds.x_max + 1j * box.bounds.y_max,
        )

        params = Parameters(
            simulation_length=5000,
            equilibrium_resolution=50,
            num_simulations=1600,
            limits=bounds,
        )

        run_simulations(params, pl, "run3", num_steps=1)

        pl.fly_to(box.center)

    pl.add_box_widget(
        callback=bounds_callback,
        rotation_enabled=False,
        bounds=(-2, 4, -2, 2, -1, 1),
    )

    # params = Parameters(
    #     simulation_length=5000,
    #     equilibrium_resolution=6,
    #     num_simulations=200,  # 800,
    #     limits=((-2 - 2j), (4 + 2j)),
    # )

    # run_simulations(params, pl, "base", num_steps=10)

    # params.limits = ((3 - 0.5j), (4 + 0.5j))
    # params.equilibrium_resolution = 50
    # run_simulations(params, pl, "run1", num_steps=3)

    # params.limits = ((3.8 - 0.1j), (3.9 + 0.1j))
    # run_simulations(params, pl, "run2", num_steps=3)

    # center = np.mean(params.limits)
    # pl.fly_to((np.real(center), np.imag(center), 1))
    # pl.fly_to((3.58355, 0, 0.9))

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
