"""Helpful code for simulating systems and updating parameters.

Functions
---------
run_update
    Iteratively simulates a `System` and updates parameter values, returning the
    sequences of parameter values and data assimilated-vs-true errors.
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
    """Use `solver` to run `system` and update parameter values with
    `optimizer`, and return sequence of parameter values and errors between
    data assimilated and true states.

    Parameters
    ----------
    system
        The system to simulate
    solver
        An instance of `base_solver.Solver` to simulate `system`
    dt
        The step size to use in `solver`
    T0
        The initial time at which to begin simulation
    Tf
        The (approximate) final time for simulation. This function will use as
        many multiples of `dt` as possible without simulating longer than `Tf`.
    t_relax
        The (approximate) length of time to simulate system between parameter
        updates. This function will use as many multiples of `dt` as possible
        without simulating longer than `t_relax` between parameter updates.
    true0
        The initial state of the true system
    assimilated0
        The initial state of the data assimilated system
    optimizer
        A callable that accepts the observed portion of the true system state
        and the data assimilated system state and returns updated `system`
        parameters.

        Note that an instance of `base_optim.Optimizer` implements this
        interface.
        If None, defaults to `base_optim.LevenbergMarquardt`.
    lr_scheduler
        Instance of `base_optim.LRScheduler` to update optimizer learning rate.
    t_begin_updates: float | None = None,
        Perform parameter updates after this time.
    return_all
        If true, return true and data assimilated states for entire simulation.

    Returns
    -------
    cs
        The sequence of parameter values
        shape (N + 1, d) where d is the number of parameters being estimated and
            N is the number of parameter updates performed (the first row is the
            initial set of parameter values).
    errors
        The sequence of errors between the true and data assimilated systems
        shape (N,) where N is the number of parameter updates performed
    tls
        The actual linspace of time values used, in multiples of `t_relax` from
        `T0` to approximately `Tf`
        shape (N + 1,) where N is the number of parameter updates performed
    true
        True states for final iteration of length `t_relax`,
        or if `return_all` is True, then true states for entire simulation.
    assimilated
        Data assimilated states for final iteration of length `t_relax`,
        or if `return_all` is True, then assimilated states for entire simulation.
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
                true[-1][system.true_observed_mask], assimilated[-1]
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
        system.cs = optimizer(
            true[-1][system.true_observed_mask], assimilated[-1]
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

    while tf <= Tf:
        true, assimilated, tls = solver.solve(
            true0,
            assimilated0,
            t0,
            tf,
            dt,
            start_with_multistep=True,
        )

        true0, assimilated0 = true[-solver.k :], assimilated[-solver.k :]

        # Update parameters
        if t_begin_updates is None or t_begin_updates <= tf:
            system.cs = optimizer(
                true[-1][system.true_observed_mask], assimilated[-1]
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
