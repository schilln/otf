"""
`run_update` iteratively simulates a `System` and updates parameter values,
returning the sequences of parameter values and nudged-vs-true errors.
"""

from collections.abc import Callable

import numpy as np
from jax import numpy as jnp

from ..optim import base as optim_base
from ..system import BaseSystem
from ..time_integration.base import (
    BaseSolver,
    MultistageSolver,
    MultistepSolver,
)

jndarray = jnp.ndarray


def run_update(
    system: BaseSystem,
    true_solver: BaseSolver,
    nudged_solver: BaseSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    nudged0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | optim_base.Optimizer
    | None = None,
) -> tuple[jndarray, np.ndarray, np.ndarray]:
    """Use `true_solver` and `nudged_solver` to run `system` and update
    parameter values with `optimizer`, and return sequence of parameter values
    and errors between nudged and true states.

    Parameters
    ----------
    system
        The system to simulate
    true_solver
        An instance of `separate_base_solver.Solver` to simulate true state of
        `system`
    nudged_solver
        An instance of `separate_base_solver.Solver` to simulate nudged state of
        `system`
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
    """
    if optimizer is None:
        optimizer = optim_base.LevenbergMarquardt(system)

    cs = [system.cs]
    errors = []

    true_args = dict()
    nudged_args = dict()

    if isinstance(true_solver, MultistepSolver):
        true_args["start_with_multistep"] = True

        # Get the initial state for the next iteration.
        get_true0 = lambda true: true[-true_solver.k :]

        # Get true states except for initial states (for error calculation).
        remove_true0 = lambda true: true[true_solver.k :]
    elif isinstance(true_solver, MultistageSolver):
        get_true0 = lambda true: true[-1]
        remove_true0 = lambda true: true[1:]
    else:
        raise NotImplementedError(
            "`true_solver` should be instance of subclass of "
            "`separate_base_solver.MultistageSolver` or "
            "`separate_base_solver.MultistepSolver`"
        )

    if isinstance(nudged_solver, MultistepSolver):
        nudged_args["start_with_multistep"] = True
        get_nudged0 = lambda nudged: nudged[-nudged_solver.k :]
        remove_nudged0 = lambda nudged: nudged[nudged_solver.k :]

        # Get true states from the previous iteration.
        get_prev_true = lambda true: true[-nudged_solver.k :]

        # Stack previous true states with current true states.
        concat_true = lambda prev_true, true: jnp.concatenate((prev_true, true))
    elif isinstance(nudged_solver, MultistageSolver):
        get_nudged0 = lambda nudged: nudged[-1]
        remove_nudged0 = lambda nudged: nudged[1:]
        get_prev_true = lambda _: None
        concat_true = lambda _, true: true
    else:
        raise NotImplementedError(
            "`nudged_solver` should be instance of subclass of "
            "`separate_base_solver.MultistageSolver` or "
            "`separate_base_solver.MultistepSolver`"
        )

    t0 = T0
    tf = t0 + t_relax

    true, tls = true_solver.solve_true(true0, t0, tf, dt)
    nudged, _ = nudged_solver.solve_nudged(
        nudged0, t0, tf, dt, true[:, system.observed_slice]
    )

    # Note: If k is -1, the extra first dimension is not eliminated.
    true0 = get_true0(true)
    nudged0 = get_nudged0(nudged)

    # Update parameters
    system.cs = optimizer(true[-1][system.observed_slice], nudged[-1])
    cs.append(system.cs)

    t0 = tls[-1]
    tf = t0 + t_relax

    # Relative error
    errors.append(
        np.linalg.norm(true[1:] - nudged[1:]) / np.linalg.norm(true[1:])
    )

    prev_true = get_prev_true(true)
    while tf <= Tf:
        true, tls = true_solver.solve_true(true0, t0, tf, dt, **true_args)
        nudged, tls = nudged_solver.solve_nudged(
            nudged0,
            t0,
            tf,
            dt,
            concat_true(prev_true, true)[:, system.observed_slice],
            **nudged_args,
        )

        true0 = get_true0(true)
        nudged0 = get_nudged0(nudged)

        # Update parameters
        system.cs = optimizer(true[-1][system.observed_slice], nudged[-1])
        cs.append(system.cs)

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(
            np.linalg.norm(remove_true0(true) - remove_nudged0(nudged))
            / np.linalg.norm(remove_true0(true))
        )
        prev_true = get_prev_true(true)

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return jnp.stack(cs), errors, tls
