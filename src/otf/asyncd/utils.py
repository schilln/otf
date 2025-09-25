"""
`run_update` iteratively simulates a `System` and updates parameter values,
returning the sequences of parameter values and data assimilated-vs-true errors.
"""

from collections.abc import Callable

import numpy as np
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
) -> tuple[jndarray, np.ndarray, np.ndarray]:
    """Use `true_solver` and `assimilated_solver` to run `system` and update
    parameter values with `optimizer`, and return sequence of parameter values
    and errors between assimilated and true states.

    Parameters
    ----------
    system
        The system to simulate
    true_observed
        Observed states of true system
        shape (N, ...)
    assimilated_solver
        An instance of `Solver` to simulate data assimilated state of `system`
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
    assimilated0
        The initial state of the data assimilated system
    optimizer
        A callable that accepts the observed portion of the true system state
        and the data assimilated system state and returns updated `system`
        parameters.

        Note that an instance of `optim.base.BaseOptimizer` implements this
        interface.
        If None, defaults to `optim.optimizer.LevenbergMarquardt`.

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
    """
    if optimizer is None:
        optimizer = opt.LevenbergMarquardt(system)

    cs = [system.cs]
    errors = []

    assimilated_args = dict()

    if isinstance(assimilated_solver, ti_base.SinglestepSolver):
        k = 1
    elif isinstance(assimilated_solver, ti_base.MultistepSolver):
        assimilated_args["start_with_multistep"] = True
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

    t0 = T0
    tf = t0 + t_relax

    assimilated, tls = assimilated_solver.solve_assimilated(
        assimilated0, t0, tf, dt, true_observed
    )
    end = len(tls)

    assimilated0 = assimilated[-k:]

    # Update parameters
    system.cs = optimizer(true_observed[end - 1], assimilated[-1])
    cs.append(system.cs)
    lr_scheduler.step()

    t0 = tls[-1]
    tf = t0 + t_relax

    # Relative error
    errors.append(
        np.linalg.norm(true_observed[1:end] - assimilated[1:])
        / np.linalg.norm(true_observed[1:end])
    )

    start = end - 1

    while tf <= Tf:
        assimilated, tls = assimilated_solver.solve_assimilated(
            assimilated0,
            t0,
            tf,
            dt,
            true_observed[start - k + 1 :],
            **assimilated_args,
        )
        end += len(tls) - k

        assimilated0 = assimilated[-k:]

        # Update parameters
        system.cs = optimizer(true_observed[end - 1], assimilated[-1])
        cs.append(system.cs)
        lr_scheduler.step()

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        errors.append(
            np.linalg.norm(true_observed[start + 1 : end] - assimilated[k:])
            / np.linalg.norm(true_observed[start + 1 : end])
        )

        start = end - 1

    errors = np.array(errors)

    # Note the last `t0` is the actual final time of the simulation.
    tls = np.linspace(T0, t0, len(errors) + 1)

    return jnp.stack(cs), errors, tls
