import numpy as np
from matplotlib import pyplot as plt


def plot(cs, u_errors, g, tls):
    num_iters = len(cs)
    ils = np.arange(num_iters)

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))

    ax = axs[0]
    ax.hlines(g, ils[0], ils[-1], label="g", color="black")
    ax.plot(ils, cs, label="c")
    ax.legend()
    ax.set_title("c vs g")
    ax.set_xlabel("Iteration number")

    ax = axs[1]
    ax.plot(tls[1:], u_errors)
    ax.set_yscale("log")
    ax.set_title("Relative error in $u$")
    ax.set_xlabel("Time")

    fig.tight_layout()

    return fig, axs
