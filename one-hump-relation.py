import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "mathtext.default": "it",
    "mathtext.fontset": "cm",
    "font.family": "Ebrima",
})

xs = np.linspace(0,1,500)
ys = xs * (1 - xs)

fig, ax = plt.subplots()


ax.plot(xs, ys)
ax.set_xlabel("$x_n$", fontsize=14)
ax.set_ylabel("$x_{n+1}$", fontsize=14)
ax.set_title(" $x_{n+1}$ vs $x_n$ in the logistic map", fontsize=14)

plt.grid(alpha=0.5, ls=':')

plt.savefig("hump_relation.png", dpi=300)
plt.show()