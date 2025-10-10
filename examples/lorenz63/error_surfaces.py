import jax
import matplotlib as mpl
import numpy as np
from jax import numpy as jnp

from otf import optim
from otf import time_integration as ti
from otf.asyncd import utils
from otf.system import base as system_base

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

jndarray = jnp.ndarray
ndarray = np.ndarray


def get_initial_values():
    # Initial true state
    u0 = jnp.array([0, 1, -1], dtype=float)

    # Initial simulation state
    un0 = jnp.zeros_like(u0)

    return u0, un0


def true_ode(gs: jndarray, true: jndarray) -> jndarray:
    sigma, rho, beta = gs

    x, y, z = true

    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def assimilated_ode(cs: jndarray, nudged: jndarray) -> jndarray:
    sigma, rho, beta = cs

    x, y, z = nudged

    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def get_errors(
    seed: int | None = 42,
) -> tuple[jndarray, ndarray, ndarray]:
    # Simulation parameters
    dt = 0.01
    T0, Tf = 0, 200
    t_relax = Tf

    # System evolution parameters
    g1, g2, g3 = 10, 28, 8 / 3
    gs = jnp.array([g1, g2, g3])

    mu = 20
    observed_slice = jnp.s_[:]

    # Set up system
    system = system_base.System_ModelKnown(
        mu, gs, None, observed_slice, assimilated_ode, true_ode
    )
    solver = ti.ForwardEuler(system)
    solver = ti.TwoStepAdamsBashforth(system, solver)

    # Solve true system
    true0, assimilated0 = get_initial_values()
    true, _ = solver.solve_true(true0, T0, Tf, dt)
    true_observed = true[:, observed_slice]

    # Select random directions in parameter values, scaling standard deviation
    # by the size of the true parameter values. Then scale each direction vector
    # to have unit length.
    rng = np.random.default_rng(seed)
    dirs = rng.normal(scale=gs, size=(2, *gs.shape))
    dirs /= np.linalg.norm(dirs, axis=1).reshape(-1, 1)

    # Set up grid of sizes of steps to take in random directions
    xn = yn = 41
    xls = np.linspace(-1, 1, xn)
    yls = np.linspace(-1, 1, yn)
    x_steps, y_steps = np.meshgrid(xls, yls)
    xis, yis = np.meshgrid(np.arange(xn), np.arange(yn))

    # Simulate on grid of parameter values
    errors = np.full((yn, xn, len(true)), np.inf)
    for x_step, y_step, xi, yi in zip(
        *map(np.ravel, (x_steps, y_steps, xis, yis))
    ):
        cs = gs + x_step * dirs[0] + y_step * dirs[1]
        system._set_cs(cs)

        # Re-initializing the optimizer is necessary when the particular
        # optimizer maintains internal state (such as Adam).
        optimizer = optim.optimizer.DummyOptimizer(system)

        _, u_errors, *_ = utils.run_update(
            system,
            true_observed,
            solver,
            dt,
            T0,
            Tf,
            t_relax,
            assimilated0,
            optimizer=optimizer,
            return_all=False,
        )

        errors[yi, xi] = u_errors

    return errors, xls, yls


def plot(fig, ax, errors: jndarray, xls: ndarray, yls: ndarray):
    cmap = mpl.cm.viridis
    cf = ax.contourf(xls, yls, errors[:, :, -1], levels=20, cmap=cmap)

    colorbar = fig.colorbar(cf, ax=ax)
    colorbar.set_label("Relative error")
