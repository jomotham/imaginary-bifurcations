from numba import njit, prange
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from dataclasses import dataclass
from pygifsicle import optimize

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
            c = 0.5 * r * (1 - 0.5 * r)
            # z_n = r * (1 / 2 - x_n)
            # z_n1 = z_n**2 + c
            # value = 1 / 2 - z_n1 / r
            # value = x_n**2 + r
            value = x_n**2 + c

            # logistic map
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
    num_simulations: tuple[int, int]
    limits: tuple[complex, complex]

    @property
    def r_real(self):
        return np.linspace(*np.real(self.limits), self.num_simulations[0])

    @property
    def r_imag(self):
        return np.linspace(*np.imag(self.limits), self.num_simulations[1])


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


def run_simulations(
    params: Parameters,
    plotter: pv.Plotter,
    id: str,
    num_steps: int,
    **kwargs,
):
    population = create_buffer(params)

    for decimation in reversed(range(0, num_steps)):
        # create a strided view of the list of coordinates
        decimated_population = population[:: 10**decimation]
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
            name=f"mymesh{id}",  # pyvista will deduplicate meshes with the same name
            **kwargs,
        )

        plotter.render()
        # plotter.write_frame() # write to the gif
        # plotter.app.processEvents()


def interactive():
    print("Hello from imaginary-bifurcations!")

    pl = BackgroundPlotter()
    pl.show_axes()
    pl.show_bounds(
        location="origin",
        xtitle="real",
        ytitle="imaginary",
        ztitle="equilibrium value",
    )
    pl.enable_parallel_projection()
    pl.show()
    # pl.open_gif("points.gif")
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
            equilibrium_resolution=10,
            num_simulations=(1500, 1500),
            limits=bounds,
        )

        run_simulations(
            params,
            pl,
            "run3",
            num_steps=1,
            cmap="gist_earth",
            opacity=0.2,
            scalars="height",
        )

        pl.fly_to(box.center)
        pl.camera.zoom(2)

    pl.add_box_widget(
        callback=bounds_callback,
        rotation_enabled=False,
        bounds=(-2, 4, -2, 2, -1, 1),
    )

    params = Parameters(
        simulation_length=8000,
        equilibrium_resolution=50,
        num_simulations=(1200, 1),
        limits=((-2 + 0), (4 + 0)),
    )
    run_simulations(params, pl, "real", num_steps=1, color="black")

    print("Yielding to eventloop")
    pl.app.exec_()
    print("Saving gif")

    pl.close()


def create_flying_gif():
    pl = pv.Plotter(off_screen=True)

    params = Parameters(
        simulation_length=8000,
        equilibrium_resolution=6,
        num_simulations=(800, 800),
        limits=((-2 - 2j), (4 + 2j)),
    )

    scalar_bar_args = dict(
        # height=0.25,
        title="color=|z|",
        title_font_size=24,
        vertical=True,
        position_y=0.3,
        # position_x=0.05,
        # position_y=0.05,
    )

    run_simulations(
        params,
        pl,
        "base",
        num_steps=1,
        cmap="gist_earth",
        opacity=0.2,
        scalar_bar_args=scalar_bar_args,
    )

    params.limits = ((-2 + 0), (4 + 0))
    params.equilibrium_resolution = 50
    params.num_simulations = (800, 1)
    run_simulations(
        params,
        pl,
        "real",
        num_steps=1,
        opacity=0.5,
        color="black",
    )

    # run_simulations(params, pl, "run1", num_steps=3)

    # params.limits = ((3.8 - 0.1j), (3.9 + 0.1j))
    # run_simulations(params, pl, "run2", num_steps=3)

    # center = np.mean(params.limits)
    # pl.fly_to((np.real(center), np.imag(center), 1))
    # pl.fly_to((3.58355, 0, 0.9))

    # pl.camera.zoom("tight")
    pl.enable_parallel_projection()
    # pl.show_axes()
    # pl.remove_scalar_bar()
    pl.show_bounds(
        location="outer",
        xtitle="real",
        ytitle="imaginary",
        ztitle="equilibrium value",
        use_3d_text=False,
    )

    print("Exporting scene")
    pl.export_html("Figures/exported_scene.html")

    print("Creating gif")

    focus = [0, 0, 0]
    point = [4, 0, 0]

    angle = np.linspace(0, np.pi / 2, num=24)
    trajectory = 1 * np.stack(
        [np.zeros_like(angle), np.cos(angle), np.sin(angle)], axis=1
    )

    viewup_trajectory = np.stack(
        [np.zeros_like(angle), -np.sin(angle), np.cos(angle)], axis=1
    )

    gif_path = "Figures/bifurcation_to_mandelbrot_tilt.gif"
    pl.open_gif(gif_path)

    pl.set_position(trajectory[0], render=False)
    pl.set_focus(focus, render=False)
    pl.set_viewup(viewup_trajectory[0])

    for _ in range(10):
        pl.write_frame()

    for point, viewup in zip(trajectory, viewup_trajectory):
        print(point)
        pl.set_position(point, render=False)
        pl.set_focus(focus, render=False)
        pl.set_viewup(viewup)
        pl.write_frame()

    for _ in range(10):
        pl.write_frame()

    pl.mwriter.close()

    print("Optimizing gif")
    optimize(gif_path)


if __name__ == "__main__":
    create_flying_gif()
    # interactive()
