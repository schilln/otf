import numpy as np
from matplotlib import pyplot as plt


def plot(cs, u_errors, gs, tls):
    num_iters, num_params = cs.shape
    ils = np.arange(num_iters)

    # Make sure there's one more axes than there are parameters (for plotting
    # the error).
    q = (num_params + 1) // 2
    fig, axs = plt.subplots(
        q if num_params % 2 == 1 else q + 1, 2, figsize=(8, 8)
    )
    # Delete last axes if unused.
    if num_params % 2 == 0:
        plt.delaxes(axs[-1, 0])

    i, j = 0, 0
    for k, c in enumerate(cs.T):
        g = gs[k] if k < len(gs) else 0

        ax = axs[i, j]
        ax.hlines(g, ils[0], ils[-1], label=f"g{k}", color="black")
        ax.plot(ils, c, label=f"c{k}")
        ax.legend()
        ax.set_title(f"c{k} vs {f'g{k}' if k < len(gs) else 0}")
        ax.set_xlabel("Iteration number")

        # Iterate through axs first from left to right, then top to bottom.
        if j == 1:
            i += 1
            j = 0
        else:
            j += 1

    ax = axs[-1, -1]
    ax.plot(tls[1:], u_errors)
    ax.set_yscale("log")
    ax.set_title("Relative error in $u$")
    ax.set_xlabel("Time")

    fig.tight_layout()

    return fig, axs
