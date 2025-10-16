import jax
import matplotlib as mpl
import numpy as np
from jax import numpy as jnp

from otf import optim
from otf.asyncd import utils
from otf.system import base as system_base
from otf.time_integration import base as ti_base

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

jndarray = jnp.ndarray
ndarray = np.ndarray


def get_errors(
    system: system_base.System_ModelKnown,
    true_observed: jndarray,
    assimilated_solver: ti_base.SinglestepSolver | ti_base.MultistepSolver,
    dt: float,
    T0: float,
    Tf: float,
    assimilated0: jndarray,
    cs_center: jndarray,
    true_actual: jndarray | None = None,
    seed: int | None = 42,
    xn: int = 11,
    yn: int = 11,
    x_relative_bound: float = 1,
    y_relative_bound: float = 1,
    start_relative_position: tuple[int, int] | None = None,
    optimizer_type: type[optim.base.BaseOptimizer] | None = None,
    optimizer_kwargs: dict = dict(),
    sim_T0: float | None = 0,
    sim_Tf: float | None = None,
    sim_t_relax: float | None = 1,
) -> dict:
    """Compute an error surface. Optionally compute a trajectory of parameter
    updates along the surface if `start_relative_position` is not None.

    Parameters
    ----------
    cs_center
        Parameter values to use as center for grid of simulations
    xn, yn
        Number of grid points to simulate system on in respective direction,
        i.e., simulate on grid of shape (xn, yn)
    x_relative_bound, y_relative_bound
        Maximum relative step size in each random direction
    start_relative_position
        If not None, used to compute starting parameter values from the error
        surface domain relative to (`x_relative_bound`, `y_relative_bound`),
        then simulate a system using `true_observed` and `optimizer` to update
        parameters. Include the sequence of coordinates of the updated
        parameters in the returned `(xls, yls)` domain.
    optimizer_type
        Optimizer type to use to update parameters if `start_relative_position`
        is not None.
    optimizer_options
        Keyword arguments with which to initialize `optimizer_type`
    sim_T0
        Initial time at which to begin simulation if `start_relative_position`
        is not None
    sim_Tf
        (Approximate) final time for simulation if `start_relative_position` is
        not None
    sim_t_relax
        (Approximate) length of time to simulate system between parameter
        updates if `start_relative_position` is not None

    Returns
    -------
    result
        Dictionary containing the following keys:

        - **errors** (*jndarray*) - Grid of relative errors, shape (yn, xn,
          len(true_observed))
        - **xls** (*ndarray*) - x-coordinates of grid, shape (xn,)
        - **yls** (*ndarray*) - y-coordinates of grid, shape (yn,)
        - **cs** (*ndarray*, optional) - Sequence of parameter values during
          optimization. Only present if `start_relative_position` is not None.
        - **cs_coordinates** (*ndarray*, optional) - Sequence of positions of
          `cs` values in `(xls, yls)` domain using updates provided by
          `optimizer`, shape (N, 2) where N is number of iterations. Only
          present if `start_relative_position` is not None.
    """
    if start_relative_position is not None:
        if optimizer_type is None:
            raise ValueError(
                "`optimizer` must be provided if `start_relative_position`"
                " is not None"
            )
        if len(start_relative_position) != 2:
            raise ValueError(
                "`start_relative_position` must contain two elements"
            )
        for item in start_relative_position:
            if not isinstance(item, (int, float)):
                raise ValueError(
                    "elements of `start_relative_position` must be numbers"
                )
        if not isinstance(sim_T0, (int, float)) or not isinstance(
            sim_Tf, (int, float)
        ):
            raise ValueError("`sim_T0` and `sim_Tf` must be numbers")

    x_relative_bound = abs(x_relative_bound)
    y_relative_bound = abs(y_relative_bound)

    # Select random directions in parameter values, scaling standard deviation
    # by the size of the true parameter values. Then scale each direction vector
    # to have unit length.
    rng = np.random.default_rng(seed)
    dirs = rng.normal(scale=cs_center, size=(2, *cs_center.shape))
    dirs /= np.linalg.norm(dirs, axis=1).reshape(-1, 1)

    # Set up grid of sizes of steps to take in random directions
    xls = np.linspace(-x_relative_bound, x_relative_bound, xn)
    yls = np.linspace(-y_relative_bound, y_relative_bound, yn)
    x_steps, y_steps = np.meshgrid(xls, yls)
    xis, yis = np.meshgrid(np.arange(xn), np.arange(yn))

    # Simulate on grid of parameter values
    errors = np.full((yn, xn, len(true_observed)), np.inf)
    for x_step, y_step, xi, yi in zip(
        *map(np.ravel, (x_steps, y_steps, xis, yis))
    ):
        system.cs = cs_center + x_step * dirs[0] + y_step * dirs[1]

        # Re-initializing the optimizer is necessary when the particular
        # optimizer maintains internal state (such as Adam).
        tmp_optimizer = optim.optimizer.DummyOptimizer(system)

        _, u_errors, *_ = utils.run_update(
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

        errors[yi, xi] = u_errors

    if start_relative_position is None:
        return {"errors": errors, "xls": xls, "yls": yls}

    system.cs = (
        cs_center
        + start_relative_position[0] * dirs[0]
        + start_relative_position[1] * dirs[1]
    )

    optimizer = optimizer_type(system, **optimizer_kwargs)
    cs, *_ = utils.run_update(
        system,
        true_observed,
        assimilated_solver,
        dt,
        sim_T0,
        sim_Tf,
        sim_t_relax,
        assimilated0,
        optimizer=optimizer,
        return_all=False,
        true_actual=true_actual,
    )
    cs_coordinates = np.linalg.lstsq(dirs.T, (cs - cs_center).T)[0].T

    return {
        "errors": errors,
        "xls": xls,
        "yls": yls,
        "cs": cs,
        "cs_coordinates": cs_coordinates,
    }


def plot(fig, ax, errors: jndarray, xls: ndarray, yls: ndarray):
    cmap = mpl.cm.viridis
    cf = ax.contourf(xls, yls, errors[:, :, -1], levels=20, cmap=cmap)

    colorbar = fig.colorbar(cf, ax=ax)
    colorbar.set_label("Relative error")
