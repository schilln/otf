"""Helpful code for simulating systems and updating parameters.

Functions
---------
run_update
    Iteratively simulates a `System` and updates parameter values, returning the
    sequences of parameter values and nudged-vs-true errors.
"""

from collections.abc import Callable

import numpy as np
from jax import numpy as jnp

import base_system
import base_solver
import base_optim

jndarray = jnp.ndarray


def run_update(
    system: base_system.System,
    solver: base_solver.Solver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | base_optim.Optimizer
    | None = None,
    lr_scheduler: base_optim.LRScheduler = base_optim.DummyLRScheduler(),
    t_begin_updates: float | None = None,
    return_all: bool = False,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use `solver` to run `system` and update parameter values with
    `optimizer`, and return sequence of parameter values and errors between
    nudged and true states.

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
    nudged0
        The initial state of the nudged system
    optimizer
        A callable that accepts the observed portion of the true system state
        and the nudged system state and returns updated `system` parameters.

        Note that an instance of `base_optim.Optimizer` implements this
        interface.
        If None, defaults to `base_optim.LevenbergMarquardt`.
    lr_scheduler
        Instance of `base_optim.LRScheduler` to update optimizer learning rate.
    t_begin_updates: float | None = None,
        Perform parameter updates after this time.
    return_all
        If true, return true and nudged states for entire simulation.

    Returns
    -------
    cs
        The sequence of parameter values
        shape (N + 1, d) where d is the number of parameters being estimated and
            N is the number of parameter updates performed (the first row is the
            initial set of parameter values).
    errors
        The sequence of errors between the true and nudged systems
        shape (N,) where N is the number of parameter updates performed
    tls
        The actual linspace of time values used, in multiples of `t_relax` from
        `T0` to approximately `Tf`
        shape (N + 1,) where N is the number of parameter updates performed
    true
        True states for final iteration of length `t_relax`,
        or if `return_all` is True, then true states for entire simulation.
    nudged
        Nudged states for final iteration of length `t_relax`,
        or if `return_all` is True, then nudged states for entire simulation.
    """
    if optimizer is None:
        optimizer = base_optim.LevenbergMarquardt(system)

    if isinstance(solver, base_solver.SinglestepSolver):
        return _run_update_singlestep(
            system,
            solver,
            dt,
            T0,
            Tf,
            t_relax,
            true0,
            nudged0,
            optimizer,
            lr_scheduler,
            t_begin_updates,
            return_all,
        )
    elif isinstance(solver, base_solver.MultistepSolver):
        return _run_update_multistep(
            system,
            solver,
            dt,
            T0,
            Tf,
            t_relax,
            true0,
            nudged0,
            optimizer,
            lr_scheduler,
            t_begin_updates,
            return_all,
        )
    else:
        raise NotImplementedError(
            "`solver` should be instance of subclass of "
            "`base_solver.SinglestepSolver` or `base_solver.MultistepSolver`"
        )


def _run_update_singlestep(
    system: base_system.System,
    solver: base_solver.SinglestepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | base_optim.Optimizer
    | None = None,
    lr_scheduler: base_optim.LRScheduler = base_optim.DummyLRScheduler,
    t_begin_updates: float | None = None,
    return_all: bool = False,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implementation of `run_update` for non-multistep solvers (e.g., RK4),
    here referred to as 'singlestep' solvers. See documentation of `run_update`.
    """
    assert isinstance(solver, base_solver.SinglestepSolver)

    if optimizer is None:
        optimizer = base_optim.LevenbergMarquardt(system)

    cs = [system.cs]
    errors = []

    if return_all:
        trues, nudgeds = (
            [np.expand_dims(true0, 0)],
            [np.expand_dims(nudged0, 0)],
        )

    t0 = T0
    tf = t0 + t_relax
    while tf <= Tf:
        true, nudged, tls = solver.solve(true0, nudged0, t0, tf, dt)

        true0, nudged0 = true[-1], nudged[-1]

        # Update parameters
        if t_begin_updates is None or t_begin_updates <= tf:
            system.cs = optimizer(true[-1][system.observed_slice], nudged[-1])
        cs.append(system.cs)
        lr_scheduler.step()

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(
            np.linalg.norm(true[1:] - nudged[1:]) / np.linalg.norm(true[1:])
        )

        if return_all:
            trues.append(true[1:])
            nudgeds.append(nudged[1:])

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return (
        jnp.stack(cs),
        errors,
        tls,
        np.concatenate(trues) if return_all else true,
        np.concatenate(nudgeds) if return_all else nudged,
    )


def _run_update_multistep(
    system: base_system.System,
    solver: base_solver.MultistepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | base_optim.Optimizer
    | None = None,
    lr_scheduler: base_optim.LRScheduler = base_optim.DummyLRScheduler,
    t_begin_updates: float | None = None,
    return_all: bool = False,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implementation of `run_update` for multistep solvers (e.g.,
    Adams–Bashforth). See documentation of `run_update`.
    """
    assert isinstance(solver, base_solver.MultistepSolver)

    if return_all:
        raise NotImplementedError("`return_all` not implemented yet")

    if optimizer is None:
        optimizer = base_optim.LevenbergMarquardt(system)

    cs = [system.cs]
    errors = []

    # First iteration
    t0 = T0
    tf = t0 + t_relax

    true, nudged, tls = solver.solve(true0, nudged0, t0, tf, dt)

    true0, nudged0 = true[-solver.k :], nudged[-solver.k :]

    # Update parameters
    if t_begin_updates is None or t_begin_updates <= tf:
        system.cs = optimizer(true[-1][system.observed_slice], nudged[-1])
    cs.append(system.cs)
    lr_scheduler.step()

    t0 = tls[-1]
    tf = t0 + t_relax

    # Relative error
    errors.append(
        np.linalg.norm(true[1:] - nudged[1:]) / np.linalg.norm(true[1:])
    )

    while tf <= Tf:
        true, nudged, tls = solver.solve(
            true0,
            nudged0,
            t0,
            tf,
            dt,
            start_with_multistep=True,
        )

        true0, nudged0 = true[-solver.k :], nudged[-solver.k :]

        # Update parameters
        if t_begin_updates is None or t_begin_updates <= tf:
            system.cs = optimizer(true[-1][system.observed_slice], nudged[-1])
        cs.append(system.cs)
        lr_scheduler.step()

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(
            np.linalg.norm(true[solver.k :] - nudged[solver.k :])
            / np.linalg.norm(true[solver.k :])
        )

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return jnp.stack(cs), errors, tls, true, nudged
