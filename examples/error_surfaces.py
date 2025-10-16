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
    x_max_relative_step: float = 1,
    y_max_relative_step: float = 1,
) -> tuple[jndarray, ndarray, ndarray]:
    """

    Parameters
    ----------
    cs_center
        Parameter values to use as center for grid of simulations
    xn, yn
        Number of grid points to simulate system on in respective direction,
        i.e., simulate on grid of shape (xn, yn)
    x_max_relative_step, y_max_relative_step
        Maximum relative step size in each random direction
    """
    x_max_relative_step = abs(x_max_relative_step)
    y_max_relative_step = abs(y_max_relative_step)

    # Select random directions in parameter values, scaling standard deviation
    # by the size of the true parameter values. Then scale each direction vector
    # to have unit length.
    rng = np.random.default_rng(seed)
    dirs = rng.normal(scale=cs_center, size=(2, *cs_center.shape))
    dirs /= np.linalg.norm(dirs, axis=1).reshape(-1, 1)

    # Set up grid of sizes of steps to take in random directions
    xls = np.linspace(-x_max_relative_step, x_max_relative_step, xn)
    yls = np.linspace(-y_max_relative_step, y_max_relative_step, yn)
    x_steps, y_steps = np.meshgrid(xls, yls)
    xis, yis = np.meshgrid(np.arange(xn), np.arange(yn))

    # Simulate on grid of parameter values
    errors = np.full((yn, xn, len(true_observed)), np.inf)
    for x_step, y_step, xi, yi in zip(
        *map(np.ravel, (x_steps, y_steps, xis, yis))
    ):
        cs = cs_center + x_step * dirs[0] + y_step * dirs[1]
        system._set_cs(cs)

        # Re-initializing the optimizer is necessary when the particular
        # optimizer maintains internal state (such as Adam).
        optimizer = optim.optimizer.DummyOptimizer(system)

        _, u_errors, *_ = utils.run_update(
            system,
            true_observed,
            assimilated_solver,
            dt,
            T0,
            Tf,
            Tf,
            assimilated0,
            optimizer=optimizer,
            return_all=False,
            true_actual=true_actual,
        )

        errors[yi, xi] = u_errors

    return errors, xls, yls


def plot(fig, ax, errors: jndarray, xls: ndarray, yls: ndarray):
    cmap = mpl.cm.viridis
    cf = ax.contourf(xls, yls, errors[:, :, -1], levels=20, cmap=cmap)

    colorbar = fig.colorbar(cf, ax=ax)
    colorbar.set_label("Relative error")
