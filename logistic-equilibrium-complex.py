import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt

# ti.init(arch=ti.gpu)
ti.init(arch=ti.cpu)

simulation_length = 200
num_simulations = 500

population = ti.field(dtype=tm.vec2, shape=(num_simulations, num_simulations, simulation_length))

arr_type = ti.types.ndarray(ndim=3)

@ti.kernel
def paint(r_values: arr_type):
    for i, j in ti.ndrange(num_simulations, num_simulations):
        r = tm.vec2(r_values[i, j, 0], r_values[i, j, 1])
        for k in range(simulation_length):
            population[i, j, k + 1] = 4 * tm.cmul(r, tm.cmul(population[i, j, k], (1 - population[i, j, k])))

@ti.kernel
def fill(real : ti.f32, imag : ti.f32):
    for i, j in ti.ndrange(num_simulations, num_simulations):
        population[i, j, 0] = tm.vec2(real, imag)

def main():
    print("Hello from imaginary-bifurcations!")

    fill(0.5, 0)

    r_re = np.linspace(0.2, 1, num_simulations, dtype=np.float32)
    r_im = np.linspace(-1, 1, num_simulations, dtype=np.float32)
    r = np.stack(np.meshgrid(r_re, r_im), axis=2)
    print("Starting calculations")
    # r = 1 - np.logspace(0, 0.8, num_simulations, dtype=np.float32)
    paint(r_values=r)
    print("Finished calculations")
    
    #ax = plt.figure().add_subplot(projection='3d')

    fig, ax = plt.subplots()

    pop = population.to_numpy()
    pop2 = pop.view(np.complex64)[..., 0]
    print(pop2.shape)
    print(pop2[:, 59, -50:])
    ax.plot(np.real(pop2[:, 59, -50:]), '.')
    plt.show()  

    # fig, ax = plt.subplots()

    # pop = population.to_numpy()
    # ax.plot(r, pop[:, -50:], ".", ms=0.1)

    # ax.set_xlabel("$r$ value")
    # ax.set_ylabel("Equilibrium value(s)")

    # plt.show()


if __name__ == "__main__":
    main()
