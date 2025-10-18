import matplotlib as mpl
import numpy as np
from jax import numpy as jnp
from matplotlib.collections import LineCollection

from otf import optim
from otf.asyncd import utils
from otf.system import base as system_base
from otf.time_integration import base as ti_base

jndarray = jnp.ndarray
ndarray = np.ndarray


def get_dirs(
    mean: ndarray,
    standard_deviation: ndarray | None = None,
    seed: int | None = None,
) -> ndarray:
    """Select random directions in parameter values.

    Select directions from normal distribution with given mean and standard
    deviation, then scale each direction vector to have unit length.

    If no standard deviation is given, use the absolute values of `mean` so that
    samples tend to differ from the mean by the same relative amounts.

    Parameters
    ----------
    mean
        Mean of normal distribution from which to sample directions

        shape (m,) where m is number of parameters
    standard_deviation
        Standard deviation of normal distribution from which to sample
        directions. If None, will default to the absolute values of `mean` or to
        one anywhere `mean` is zero.

        shape (m,) where m is number of parameters
    seed
        Seed with which to initialize random number generator. If None, no fixed
        seed will be used, so results will vary from call to call.

    Returns
    -------
    dirs
        Direction vectors which may be used to define domain on which to compute
        and plot error surface

        shape (2, m) where m is number of parameters
    """
    if standard_deviation is None:
        standard_deviation = abs(mean)
        standard_deviation = np.where(
            standard_deviation == 0, 1, standard_deviation
        )
    elif mean.shape != standard_deviation.shape:
        raise ValueError("`mean` and `standard_deviation` must have same shape")

    rng = np.random.default_rng(seed)
    dirs = rng.normal(scale=standard_deviation, size=(2, *mean.shape))
    dirs /= np.linalg.norm(dirs, axis=1).reshape(-1, 1)

    return dirs


def get_surface(
    system: system_base.System_ModelKnown,
    true_observed: jndarray,
    assimilated_solver: ti_base.SinglestepSolver | ti_base.MultistepSolver,
    dt: float,
    T0: float,
    Tf: float,
    assimilated0: jndarray,
    cs_center: jndarray,
    dirs: ndarray,
    true_actual: jndarray | None = None,
    xn: int = 11,
    yn: int = 11,
    x_relative_bound: float | tuple[float, float] = 1,
    y_relative_bound: float | tuple[float, float] = 1,
) -> tuple[ndarray, ndarray, ndarray]:
    """Compute an error surface.

    Parameters
    ----------
    cs_center
        Parameter values to use as center for grid of simulations

        shape (m,) where m is number of parameters
    dirs
        Direction vectors defining domain for computing error surface

        shape (2, m) where m is number of parameters
    xn, yn
        Number of grid points to simulate system on in respective direction,
        i.e., simulate on grid of shape (xn, yn)
    x_relative_bound, y_relative_bound
        Maximum relative step size in each random direction

    Returns
    -------
    errors
        Grid of relative errors, shape (yn, xn)
    xls, yls
        Grid of coordinates relative to `dirs`, shapes (xn,) and (yn,)

    Notes
    -----
    This calls `utils.run_update(..., t_relax=Tf)` to effectively disable
    parameter updates while computing the surface.
    """
    if cs_center.ndim != 1:
        raise ValueError("`cs_center` should be one-dimensional")

    m = cs_center.shape[0]
    if dirs.ndim != 2 or dirs.shape != (2, m):
        raise ValueError("`dirs` must have shape (2, m)")

    if isinstance(x_relative_bound, float | int):
        x_relative_bound = abs(x_relative_bound)
        x_relative_bound = (-x_relative_bound, x_relative_bound)
    if isinstance(y_relative_bound, float | int):
        y_relative_bound = abs(y_relative_bound)
        y_relative_bound = (-y_relative_bound, y_relative_bound)

    # Set up grid of sizes of steps to take in random directions
    xls = np.linspace(*x_relative_bound, xn)
    yls = np.linspace(*y_relative_bound, yn)
    x_steps, y_steps = np.meshgrid(xls, yls)
    xis, yis = np.meshgrid(np.arange(xn), np.arange(yn))

    if system.cs is not None:
        original_cs = system.cs.copy()
    else:
        original_cs = None
    try:
        # Simulate on grid of parameter values
        errors = np.full((yn, xn), np.inf)
        for x_step, y_step, xi, yi in zip(
            *map(np.ravel, (x_steps, y_steps, xis, yis))
        ):
            step = (x_step, y_step)
            system.cs = get_cs_from_relative_position(cs_center, dirs, step)

            # Re-initializing the optimizer is necessary when the particular
            # optimizer maintains internal state (such as Adam).
            tmp_optimizer = optim.optimizer.DummyOptimizer(system)

            _, u_error, *_ = utils.run_update(
                system,
                true_observed,
                assimilated_solver,
                dt,
                T0,
                Tf,
                Tf,
                assimilated0,
                optimizer=tmp_optimizer,
                return_all=False,
                true_actual=true_actual,
            )
            errors[yi, xi] = u_error
    finally:
        system.cs = original_cs

    return errors, xls, yls


def get_trajectory(
    system: system_base.System_ModelKnown,
    true_observed: jndarray,
    assimilated_solver: ti_base.SinglestepSolver | ti_base.MultistepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    assimilated0: jndarray,
    optimizer: optim.base.BaseOptimizer,
    cs_center: jndarray,
    dirs: ndarray,
    run_update_options: dict = dict(),
) -> tuple[ndarray, ndarray]:
    """Compute sequence of parameter values relative to `dirs` during one
    simulation of system with updates provided by `optimizer`.

    Returns
    -------
    cs
        The sequence of parameter values

        shape (N + 1, d) where d is the number of parameters being estimated and
        N is the number of parameter updates performed (the first row is the
        initial set of parameter values).
    cs_coordinates
        Sequence of positions of `cs` values relative to `dirs`

        shape (N + 1, 2) where N is number of iterations
    """
    cs, *_ = utils.run_update(
        system,
        true_observed,
        assimilated_solver,
        dt,
        T0,
        Tf,
        t_relax,
        assimilated0,
        optimizer=optimizer,
        return_all=False,
        **run_update_options,
    )
    return cs, get_relative_position_from_cs(cs_center, dirs, cs)


def get_cs_from_relative_position(
    cs_center: ndarray,
    dirs: ndarray,
    relative_position: tuple[float, float] | ndarray,
) -> ndarray:
    """Get actual parameter values relative to `cs_center` and `dirs`.

    Parameters
    ----------
    cs_center
        Parameter values to use as center for grid of simulations

        shape (m,) where m is number of parameters
    dirs
        Direction vectors defining domain for computing error surface

        shape (2, m) where m is number of parameters
    relative_position
        Pair of coordinates in `(dirs[0], dirs[1])` basis with `cs_center` as
        the origin

    Returns
    -------
    cs
        Parameter values

        shape (m,) where m is number of parameters
    """
    if not isinstance(relative_position, ndarray):
        relative_position = np.array(relative_position)
    if relative_position.shape != (2,):
        raise ValueError("`relative_position` should have shape (2,)")

    return cs_center + relative_position @ dirs


def get_relative_position_from_cs(
    cs_center: ndarray,
    dirs: ndarray,
    cs: ndarray,
) -> ndarray:
    """Get position relative to `cs_center` and `dirs`.

    Parameters
    ----------
    cs_center
        Parameter values to use as center for grid of simulations

        shape (m,) where m is number of parameters
    dirs
        Direction vectors defining domain for computing error surface

        shape (2, m) where m is number of parameters
    cs
        Parameter values

        shape (N, m) where N is number of iterations and m is number of
        parameters

    Returns
    -------
    cs_coordinates
        Coordinates in `(dirs[0], dirs[1])` basis with `cs_center` as the origin

        shape (N, 2) where N is number of iterations and m is number of
        parameters
    """
    if cs_center.ndim != 1:
        raise ValueError("`cs_center` should be one-dimensional")

    m = cs_center.shape[0]
    if dirs.ndim != 2 or dirs.shape != (2, m):
        raise ValueError("`dirs` must have shape (2, m)")

    if cs.ndim > 2:
        raise ValueError("`cs.ndim` is greater than two")

    if cs.ndim == 1:
        cs = np.expand_dims(cs, 0)

    cs_coordinates = np.linalg.lstsq(dirs.T, (cs - cs_center).T)[0].T

    return (
        cs_coordinates.squeeze(0)
        if cs_coordinates.shape[0] == 1
        else cs_coordinates
    )


def plot_surface(
    fig, ax, errors: ndarray, xls: ndarray, yls: ndarray, levels: int = 20
):
    cmap = mpl.cm.viridis

    cf = ax.contourf(xls, yls, errors, levels=levels, cmap=cmap)

    fig.colorbar(cf, ax=ax, label="Relative error")


def plot_trajectory(fig, ax, cs_coordinates: ndarray):
    cmap = mpl.cm.magma
    xs, ys = cs_coordinates.T

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cls = np.arange(len(xs))
    norm = mpl.colors.Normalize(cls[0], cls[-1])

    lc = LineCollection(
        segments, cmap=cmap, norm=norm, linewidth=2, capstyle="round"
    )
    lc.set_array(cls)
    ax.add_collection(lc)

    ax.scatter(xs[0], ys[0], color="red", alpha=0.8, s=100, zorder=2)

    fig.colorbar(lc, ax=ax, label="Parameter update step")
