"""Utilities for running synchronized simulations and parameter updates.

This module provides helpers to run a `System` forward in time using a
`time_integration.BaseSolver` and to perform periodic parameter updates (for
example via a callable optimizer or an instance of `optim.base.BaseOptimizer`).
The primary public API is `run_update` which returns the parameter history,
relative errors, the times of updates, and the simulated true and assimilated
trajectories (either for the final relaxation window or for the whole simulation
when `return_all=True`).
"""

from collections.abc import Callable

import numpy as np
from jax import numpy as jnp

from ..optim import base as optim_base
from ..optim import lr_scheduler
from ..optim import optimizer as opt
from ..system import System_ModelKnown
from ..time_integration import base as ti_base

jndarray = jnp.ndarray


def run_update(
    system: System_ModelKnown,
    solver: ti_base.BaseSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    assimilated0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | optim_base.BaseOptimizer
    | None = None,
    lr_scheduler: lr_scheduler.LRScheduler = lr_scheduler.DummyLRScheduler(),
    t_begin_updates: float | None = None,
    return_all: bool = False,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run `system` forward and perform periodic parameter updates.

    The function advances `system` from time `T0` toward `Tf` in blocks of
    approximately `t_relax` (rounded to multiples of `dt`) using `solver`. After
    each block it updates `system.cs` using `optimizer` (a callable or a
    `optim.base.BaseOptimizer`), optionally adjusting the learning rate via
    `lr_scheduler`.

    Parameters
    ----------
    system
        The system to simulate.
    solver
        A `time_integration.BaseSolver` instance for stepping the dynamics.
    dt
        Time-step size passed to `solver`.
    T0, Tf
        Initial and (approximate) final times for the simulation.
    t_relax
        Approximate duration between parameter updates.
    true0, assimilated0
        Initial states for the true and assimilated systems (arrays compatible
        with `solver`).
    optimizer
        Callable or `optim.base.BaseOptimizer` used to compute the next
        `system.cs` from observed portions of the trajectories. If None a
        default `opt.LevenbergMarquardt` is used.
    lr_scheduler
        An `lr_scheduler.LRScheduler` instance to step each update (defaults to
        a dummy no-op scheduler).
    t_begin_updates
        If provided, updates are skipped until simulation time exceeds this
        value.
    return_all
        When True, return the full simulated trajectories for all time blocks;
        otherwise return only the last block's trajectories.

    Returns
    -------
    tuple
        `(cs, errors, tls, true, assimilated)` where
            - `cs` is an array of parameter vectors, shape `(N+1, d)`;
            - `errors` is a 1-D array of relative errors, shape `(N,)`;
            - `tls` is the time array for update times, shape `(N+1,)`;
            - `true` and `assimilated` are the final true assimilated states for
              the last interval or the full concatenated states if `return_all`
              is True.
    """
    if optimizer is None:
        optimizer = opt.LevenbergMarquardt(system)

    if isinstance(solver, (ti_base.SinglestepSolver, ti_base.MultistageSolver)):
        return _run_update_not_multistep(
            system,
            solver,
            dt,
            T0,
            Tf,
            t_relax,
            true0,
            assimilated0,
            optimizer,
            lr_scheduler,
            t_begin_updates,
            return_all,
        )
    elif isinstance(solver, ti_base.MultistepSolver):
        return _run_update_multistep(
            system,
            solver,
            dt,
            T0,
            Tf,
            t_relax,
            true0,
            assimilated0,
            optimizer,
            lr_scheduler,
            t_begin_updates,
            return_all,
        )
    else:
        raise NotImplementedError(
            "`solver` should be instance of subclass of "
            "`SinglestepSolver`, `MultistageSolver` or `MultistepSolver`"
        )


def _run_update_not_multistep(
    system: System_ModelKnown,
    solver: ti_base.MultistageSolver | ti_base.SinglestepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    assimilated0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | optim_base.BaseOptimizer
    | None = None,
    lr_scheduler: lr_scheduler.LRScheduler = lr_scheduler.DummyLRScheduler,
    t_begin_updates: float | None = None,
    return_all: bool = False,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implementation of `run_update` for non-multistep solvers (e.g., RK4),
    here referred to as 'singlestep' solvers. See documentation of `run_update`.
    """
    assert isinstance(
        solver, (ti_base.SinglestepSolver, ti_base.MultistageSolver)
    )

    if optimizer is None:
        optimizer = opt.LevenbergMarquardt(system)

    cs = [system.cs]
    errors = []

    if return_all:
        trues, assimilateds = (
            [np.expand_dims(true0, 0)],
            [np.expand_dims(assimilated0, 0)],
        )

    t0 = T0
    tf = t0 + t_relax
    while tf <= Tf:
        true, assimilated, tls = solver.solve(true0, assimilated0, t0, tf, dt)

        true0, assimilated0 = true[-1], assimilated[-1]

        # Update parameters
        if t_begin_updates is None or t_begin_updates <= tf:
            system.cs = optimizer(
                true[:, system.true_observed_mask], assimilated
            )
            lr_scheduler.step()
        cs.append(system.cs)

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(
            np.linalg.norm(
                true[1:, system.true_observed_mask]
                - assimilated[1:, system.observed_mask]
            )
            / np.linalg.norm(true[1:, system.true_observed_mask])
        )

        if return_all:
            trues.append(true[1:])
            assimilateds.append(assimilated[1:])

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return (
        jnp.stack(cs),
        errors,
        tls,
        np.concatenate(trues) if return_all else true,
        np.concatenate(assimilateds) if return_all else assimilated,
    )


def _run_update_multistep(
    system: System_ModelKnown,
    solver: ti_base.MultistepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    assimilated0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | optim_base.BaseOptimizer
    | None = None,
    lr_scheduler: lr_scheduler.LRScheduler = lr_scheduler.DummyLRScheduler,
    t_begin_updates: float | None = None,
    return_all: bool = False,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implementation of `run_update` for multistep solvers (e.g.,
    Adams–Bashforth). See documentation of `run_update`.
    """
    assert isinstance(solver, ti_base.MultistepSolver)

    if optimizer is None:
        optimizer = opt.LevenbergMarquardt(system)

    cs = [system.cs]
    errors = []

    if return_all:
        trues, assimilateds = (
            [np.expand_dims(true0, 0)],
            [np.expand_dims(assimilated0, 0)],
        )

    # First iteration
    t0 = T0
    tf = t0 + t_relax

    true, assimilated, tls = solver.solve(true0, assimilated0, t0, tf, dt)

    if return_all:
        trues.append(true[1:])
        assimilateds.append(assimilated[1:])

    true0, assimilated0 = true[-solver.k :], assimilated[-solver.k :]

    # Update parameters
    if t_begin_updates is None or t_begin_updates <= tf:
        system.cs = optimizer(true[:, system.true_observed_mask], assimilated)
        lr_scheduler.step()
    cs.append(system.cs)

    t0 = tls[-1]
    tf = t0 + t_relax

    # Relative error
    errors.append(
        np.linalg.norm(
            true[1:, system.true_observed_mask]
            - assimilated[1:, system.observed_mask]
        )
        / np.linalg.norm(true[1:, system.true_observed_mask])
    )

    while tf <= Tf:
        true, assimilated, tls = solver.solve(true0, assimilated0, t0, tf, dt)

        true0, assimilated0 = true[-solver.k :], assimilated[-solver.k :]

        # Update parameters
        if t_begin_updates is None or t_begin_updates <= tf:
            system.cs = optimizer(
                true[:, system.true_observed_mask], assimilated
            )
            lr_scheduler.step()
        cs.append(system.cs)

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(
            np.linalg.norm(
                true[solver.k :, system.true_observed_mask]
                - assimilated[solver.k :, system.observed_mask]
            )
            / np.linalg.norm(true[solver.k :, system.true_observed_mask])
        )

        if return_all:
            trues.append(true[solver.k :])
            assimilateds.append(assimilated[solver.k :])

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return (
        jnp.stack(cs),
        errors,
        tls,
        np.concatenate(trues) if return_all else true,
        np.concatenate(assimilateds) if return_all else assimilated,
    )
