from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ti.init(arch=ti.gpu)

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "newcent",
# })

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
        r = r_real[i]
        population[i, j, :] = r_imag[j]  # set initial condition
        for k in range(simulation_length):
            size = equilibrium_resolution
            population[i, j, (k + 1) % size] = (
                4 * r * population[i, j, k % size] * (1 - population[i, j, k % size])
            )

    return population


def main():
    print("Hello from imaginary-bifurcations!")

    population[..., 0] = 0.5
    r_re = np.linspace(0.2, 1, num_simulations)
    r_im = np.linspace(0, 1, num_simulations)
    # print(r.shape, r)
    print("Yay")
    # r = 1 - np.logspace(0, 0.8, num_simulations, dtype=np.float32)
    paint(population, r_re, r_im)
    print("Finished calculations")

    fig, ax = plt.subplots()
    # ax = plt.figure().add_subplot(projection="3d")

    # print(population[:, 450, 0])

    # for index in np.ndindex(population.shape[:2]):
    #     samples = population[index]
    #     # print(index, samples.shape)
    #     real = r_re[index[0]] * np.ones_like(samples)
    #     imag = np.real(r_im[index[1]]) * np.ones_like(samples)
    #     ax.plot(real, imag, np.abs(samples), ".", ms=2)
    # ax.plot(np.abs(population[:, 250, 1]), ".")
    lines = ax.plot(
        r_re, np.abs(population[:, num_simulations // 2, :]), ".", ms=0.5, c="black"
    )
    ax.set_xlabel(r"$\mathrm{Re}(r)$")
    ax.set_ylabel("Equilibrium value")

    def frame(n: int):
        print(f"{round(n/num_simulations * 100, 1)}%")
        for i, line in enumerate(lines):
            line.set_ydata(np.abs(population[:, n, i]))
            line.set_xdata(r_re)
        ax.autoscale(True)
        title = ax.set_title(
            f"$x_0$ = {'{:.4f}'.format(r_im[n])}"
        )  # update the plot title
        return (line, title)  # return the Artists that have changed

    ani = FuncAnimation(fig, frame, range(1, num_simulations), interval=100, blit=True)
    ani.save("slice_through_initial_condition.gif", fps=30, dpi=100)

    # ax.plot(np.abs(population[:, 306]), ".", ms=2)

    # plt.show()


if __name__ == "__main__":
    main()
