"""Utilities for asynchronous assimilation runs.

This module provides `run_update`, which simulates a `BaseSystem`, performs
parameter updates at regular intervals using an optimizer, and returns the
parameter trajectory together with error statistics between assimilated and
true states.
"""

from collections.abc import Callable

import numpy as np
import scipy
from jax import numpy as jnp

from ..optim import base as optim_base
from ..optim import lr_scheduler
from ..optim import optimizer as opt
from ..system import BaseSystem
from ..time_integration import base as ti_base

jndarray = jnp.ndarray


def run_update(
    system: BaseSystem,
    true_observed: jndarray,
    assimilated_solver: ti_base.SinglestepSolver | ti_base.MultistepSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    assimilated0: jndarray,
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | optim_base.BaseOptimizer
    | None = None,
    lr_scheduler: lr_scheduler.LRScheduler = lr_scheduler.DummyLRScheduler(),
    t_begin_updates: float | None = None,
    return_all: bool = False,
    true_actual: jndarray | None = None,
    weight: jndarray | None = None,
    num_loops: int = 1,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run an assimilation loop and update system parameters.

    Simulate `system` using `assimilated_solver` while comparing to
    `true_observed`, perform parameter updates using `optimizer` at each
    relaxation interval, and return parameter trajectories and relative errors.

    Parameters
    ----------
    system
        The system to simulate.
    true_observed
        Observed states of the true system, shape (K, n) where K is the number
        of observations and n is number of dimensions in the observations.
    assimilated_solver
        Solver used to advance the assimilated state.
    dt
        Time step used by the solver.
    T0, Tf
        Start and (approximate) end times for the simulation.
    t_relax
        Time between parameter updates.
    assimilated0
        Initial assimilated state.
    optimizer
        Callable accepting `(true_obs, assimilated)` and returning new
        `system.cs`. If `None`, defaults to `opt.LevenbergMarquardt(system)`.
    lr_scheduler
        Scheduler to update optimizer learning rate.
    t_begin_updates
        Time after which updates begin. If `None`, updates start immediately.
    return_all
        If True, return assimilated states for the entire simulation.
    true_actual
        If provided, used for error computation instead of `true_observed`.
    weight
        Positive-definite matrix used to weight the error norm.
    num_loops
        Number of optimizer steps per update interval.

    Returns
    -------
    tuple
        `(cs, errors, tls, assimilated)` where
            - `cs` is an array of parameter vectors, shape `(N+1, d)`;
            - `errors` is a 1-D array of relative errors, shape `(N,)`;
            - `tls` is the time array for update times, shape `(N+1,)`;
            - `assimilated` is the final assimilated states for the last
                interval or the full concatenated states if `return_all` is
                True.
    """
    if optimizer is None:
        optimizer = opt.LevenbergMarquardt(system)

    if assimilated0.ndim == 1:
        assimilated0 = jnp.expand_dims(assimilated0, 0)

    cs = [system.cs]
    errors = []

    if isinstance(assimilated_solver, ti_base.SinglestepSolver):
        k = 1
    elif isinstance(assimilated_solver, ti_base.MultistepSolver):
        k = assimilated_solver.k

        if assimilated_solver.uses_multistage:
            raise NotImplementedError(
                "`assimilated_solver` depends on a `MultistageSolver` through"
                " its `pre_multistep_solver`; all pre-multistep solvers should"
                " be singlestep or multistep"
            )
    elif isinstance(assimilated_solver, ti_base.MultistageSolver):
        raise NotImplementedError(
            "`MultistageSolver` not yet supported for `assimilated_solver`;"
            " should be instance of subclass of `MultistepSolver`"
        )
    else:
        raise NotImplementedError(
            "`assimilated_solver` should be instance of subclass of"
            " `SinglestepSolver` or `MultistepSolver`"
        )

    if return_all:
        assimilateds = [assimilated0]

    if weight is None:
        norm = np.linalg.norm
    if weight is not None:
        sqrt_weight = scipy.linalg.sqrtm(weight)
        norm = lambda states, *args, **kwargs: np.linalg.norm(
            sqrt_weight @ states.T, *args, **kwargs
        )

    t0 = T0
    tf = t0 + t_relax

    start = len0 = len(assimilated0)

    num_steps = assimilated_solver.compute_num_steps(t0, tf, dt) - len0
    end = len0 + num_steps
    assimilated, tls = assimilated_solver.solve_assimilated(
        assimilated0, t0, tf, dt, true_observed[:end]
    )

    if return_all:
        assimilateds.append(assimilated[len0:])

    assimilated0 = assimilated[-k:]

    # Update parameters
    if t_begin_updates is None or t_begin_updates <= tf:
        system.cs = optimizer(true_observed[start:end], assimilated[start:])
        lr_scheduler.step()
    cs.append(system.cs)

    t0 = tls[-1]
    tf = t0 + t_relax

    if true_actual is not None:
        true_compare = true_actual

        def assimilated_compare(assimilated):
            return assimilated
    else:
        true_compare = true_observed

        def assimilated_compare(assimilated):
            return assimilated[:, system.observed_mask]

    # Relative error
    errors.append(
        norm(true_compare[start:end] - assimilated_compare(assimilated[start:]))
        / norm(true_compare[start:end])
    )

    start = end

    while tf <= Tf:
        num_steps = assimilated_solver.compute_num_steps(t0, tf, dt) - 1
        end += num_steps

        if t_begin_updates is None or t_begin_updates <= tf:
            iters = num_loops
        else:
            iters = 1

        for _ in range(iters):
            assimilated, tls = assimilated_solver.solve_assimilated(
                assimilated0,
                t0 - dt * (k - 1),
                tf,
                dt,
                true_observed[start - k : end],
            )

            # Update parameters
            if t_begin_updates is None or t_begin_updates <= tf:
                system.cs = optimizer(true_observed[start:end], assimilated[k:])
                lr_scheduler.step()
            cs.append(system.cs)

        if return_all:
            assimilateds.append(assimilated[k:])

        assimilated0 = assimilated[-k:]

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(
            norm(true_compare[start:end] - assimilated_compare(assimilated[k:]))
            / norm(true_compare[start:end])
        )

        start = end

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return (
        jnp.stack(cs),
        errors,
        tls,
        np.concatenate(assimilateds) if return_all else assimilated,
    )
