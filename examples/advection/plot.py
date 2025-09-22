import numpy as np
from jax import numpy as jnp
from matplotlib import cm, colors
from matplotlib import pyplot as plt

jndarray = jnp.ndarray


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


def plot_3d(
    data: jndarray, T0: float, Tf: float, dt: float, x0: float, xf: float
):
    """Create a surface plot.

    Parameters
    ----------
    data
        True or nudged states over a time period
        shape (number of time steps, number of spatial steps)
    T0, Tf
        Initial and final time values
    dt
        Time step between states
    x0, xf
        Endpoints of spatial domain
    """
    tn, xn = data.shape
    xls = jnp.linspace(x0, xf, xn)
    yls = jnp.arange(tn) * dt

    X, Y = jnp.meshgrid(xls, yls)

    zlim = (np.nanmin(data), np.nanmax(data))
    cmap = cm.viridis
    norm = colors.Normalize(*zlim)

    fig, ax = plt.subplots(
        1, 1, figsize=(10, 8), subplot_kw={"projection": "3d"}
    )
    ax.view_init(30, -75)

    ax.set_xlim(x0, xf)
    ax.set_ylim(T0, Tf)
    ax.set_zlim(*zlim)

    ax.plot_surface(X, Y, data, cmap=cmap)
    fig.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, label="$u(x, t)$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_zlabel("$u(x, t)$")

    fig.suptitle(rf"$u(x, t), \quad x \in [{T0}, {Tf}]$")
    fig.tight_layout()
    plt.show()


def plot_contour(
    data: jndarray, T0: float, Tf: float, dt: float, x0: float, xf: float
):
    """Create a contour plot.

    Parameters
    ----------
    data
        True or nudged states over a time period
        shape (number of time steps, number of spatial steps)
    T0, Tf
        Initial and final time values
    dt
        Time step between states
    x0, xf
        Endpoints of spatial domain
    """
    tn, xn = data.shape
    xls = jnp.linspace(x0, xf, xn)
    yls = jnp.arange(tn) * dt

    X, Y = jnp.meshgrid(xls, yls)

    zlim = (np.nanmin(data), np.nanmax(data))
    cmap = cm.viridis
    norm = colors.Normalize(*zlim)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.set_xlim(x0, xf)
    ax.set_ylim(T0, Tf)

    ax.contourf(X, Y, data, cmap=cmap, extend="both")
    fig.colorbar(cm.ScalarMappable(norm, cmap), ax=ax, label="$u(x, t)$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    fig.tight_layout()
    plt.show()
