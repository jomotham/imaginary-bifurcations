from numba import njit, prange
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "newcent",
})


n = 100
frame_max = 300
population = np.zeros(dtype=np.float32, shape=(n,))

def paint(r: float):
    for _ in range(1):
        for i in range(n - 1):
            population[i + 1] = 4 * r * population[i] * (1 - population[i])


def main():
    print("Hello from imaginary-bifurcations!")

    fig, ax = plt.subplots()

    (line,) = ax.plot(np.zeros_like(population), ".-")
    ax.set_ylim((0,1))
    ax.set_xlabel(r"\bf{Generation}")
    ax.set_ylabel(r"\bf{Population Proportion}")
    
    def frame(n: float):
        print(n)
        r = n / frame_max
        population[0] = 0.5
        paint(r)
        line.set_ydata(population)
        title = ax.set_title(f"Growth rate r = {'{:.3f}'.format(round(r,3))}")
        return (line, title)

    ani = FuncAnimation(fig, frame, range(frame_max // 2, frame_max), interval=200)
    ani.save("logi.gif", fps=30, dpi=100)


if __name__ == "__main__":
    main()
