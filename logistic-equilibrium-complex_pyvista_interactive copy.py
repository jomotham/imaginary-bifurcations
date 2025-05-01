from numba import njit, prange
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from dataclasses import dataclass


@njit(parallel=True)
def paint(population, simulation_length):
    for idx in prange(population.shape[0]):
        r = 4 * (population[idx, 0, 0] + 1j * population[idx, 1, 0])
        for k in range(simulation_length):
            size = population.shape[2]
            value = (
                r * population[idx, 2, k % size] * (1 - population[idx, 2, k % size])
            )
            population[idx, 2, (k + 1) % size] = value

    return population


@dataclass
class Parameters:
    simulation_length: int
    equilibrium_resolution: int
    num_simulations: int

    __r: tuple[np.ndarray, np.ndarray] = None

    @property
    def r(self) -> tuple[np.ndarray, np.ndarray]:
        return self.__r


def create_buffer(params: Parameters):
    r_re, r_im = params.r

    grid_re, grid_im, grid_pop = np.meshgrid(
        r_re, r_im, np.zeros(shape=(params.equilibrium_resolution,), dtype=np.complex64)
    )

    population = np.stack([grid_re, grid_im, grid_pop], axis=2)
    population = np.reshape(population, shape=(-1, 3, params.equilibrium_resolution))

    print(population.shape)

    return population


def create_mesh(params: Parameters, population: np.ndarray):
    coords = np.reshape(population, (-1, 3))
    # check that we haven't made a copy
    assert coords.base is not None

    real_coords = coords.view(np.float64)
    strided_view = real_coords[:, 0:-1:2]
    print(real_coords.shape, strided_view.shape)
    abs_coords = np.real(coords)
    print(abs_coords.shape)

    phases = np.angle(coords[:, 2])

    print("Making mesh")
    mesh = pv.PolyData(abs_coords)
    # mesh.point_data["height"] = coords[:, 2]
    mesh.point_data["angles"] = phases

    return mesh


def main():
    print("Hello from imaginary-bifurcations!")

    params = Parameters(
        simulation_length=5000,
        equilibrium_resolution=6,
        num_simulations=10,
    )

    params._Parameters__r = (
        np.linspace(-0.5, 1, params.num_simulations),
        np.linspace(-0.5, 0.5, params.num_simulations),
    )
    # print(params)

    population = create_buffer(params)

    r_re, r_im = params.r
    print("Yay")
    # paint(population, r_re, r_im, params.simulation_length)
    print("Finished calculations")

    # population = simulate(params)
    mesh = create_mesh(params, population)

    pl = pv.Plotter()
    pl.show_axes()

    print("Inserting mesh")
    pl.add_mesh(mesh, scalars="angles", name="mymesh")
    # pl.show_bounds()
    pl.show()

    # print("Creating gif")
    # viewup = [0.5, 0.5, 1]
    # path = pl.generate_orbital_path(
    #     factor=2.0, shift=mesh.length, viewup=viewup, n_points=36
    # )
    # pl.open_gif("orbit.gif")
    # pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
    # pl.close()


if __name__ == "__main__":
    main()
