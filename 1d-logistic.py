import taichi as ti
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)

n = 50
population = ti.field(dtype=ti.f32, shape=(n,))


@ti.kernel
def paint(r: float):
    for _ in range(1):
        for i in range(n):
            population[i + 1] = 4 * r * population[i] * (1 - population[i])


def main():
    print("Hello from imaginary-bifurcations!")

    population[0] = 0.1
    paint(0.7)

    fig, ax = plt.subplots()

    ax.plot(population.to_numpy(), ".")

    plt.show()


if __name__ == "__main__":
    main()
