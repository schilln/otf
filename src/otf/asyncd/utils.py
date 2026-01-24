"""
`run_update` iteratively simulates a `System` and updates parameter values,
returning the sequences of parameter values and data assimilated-vs-true errors.
"""

from collections.abc import Callable
from enum import Enum

import jax
import numpy as np
import scipy
from jax import numpy as jnp

from ..optim import base as optim_base
from ..optim import lr_scheduler
from ..optim import optimizer as opt
from ..system import BaseSystem
from ..time_integration import base as ti_base

jndarray = jnp.ndarray


class ParameterUpdateOption(Enum):
    last_state = 0
    mean_state = 1
    mean_gradient = 2


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
    parameter_update_option: ParameterUpdateOption = ParameterUpdateOption.last_state,
    weight: jndarray | None = None,
) -> tuple[jndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    lr_scheduler
        Instance of `base_optim.LRScheduler` to update optimizer learning rate.
    t_begin_updates
        Perform parameter updates after this time.
    return_all
        If true, return data assimilated states for entire simulation.
    true_actual
        Actual states of true system. If provided, used to compute error in
        place of `true_observed`. Useful if `true_observed` contains noise.
        shape (N, ...)
    weight
        Weight the error using a (positive definite) matrix.

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
    assimilated
        Data assimilated states for final iteration of length `t_relax`, or if
        `return_all` is True, then assimilated states for entire simulation.
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

    match parameter_update_option:
        case ParameterUpdateOption.last_state:
            update = update_last_state
        case ParameterUpdateOption.mean_state:
            update = update_mean_state
        case ParameterUpdateOption.mean_gradient:
            update = update_mean_derivative

    if weight is None:
        norm = np.linalg.norm
    if weight is not None:
        sqrt_weight = scipy.linalg.sqrtm(weight)
        norm = lambda states, *args, **kwargs: np.linalg.norm(
            sqrt_weight @ states.T, *args, **kwargs
        )

    t0 = T0
    tf = t0 + t_relax

    len0 = len(assimilated0)
    num_steps = assimilated_solver.compute_num_steps(t0, tf, dt) - 1
    end = len0 + num_steps
    assimilated, tls = assimilated_solver.solve_assimilated(
        assimilated0, t0, tf, dt, true_observed[:end]
    )

    if return_all:
        assimilateds.append(assimilated[len0:])

    assimilated0 = assimilated[-k:]

    # Update parameters
    if t_begin_updates is None or t_begin_updates <= tf:
        system.cs = update(optimizer, true_observed, assimilated, 0, end, 1)
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
        norm(true_compare[len0:end] - assimilated_compare(assimilated[len0:]))
        / norm(true_compare[len0:end])
    )

    start = end

    while tf <= Tf:
        num_steps = assimilated_solver.compute_num_steps(t0, tf, dt) - 1
        end += num_steps
        assimilated, tls = assimilated_solver.solve_assimilated(
            assimilated0, t0, tf, dt, true_observed[start - k : end]
        )

        if return_all:
            assimilateds.append(assimilated[k:])

        assimilated0 = assimilated[-k:]

        # Update parameters
        if t_begin_updates is None or t_begin_updates <= tf:
            system.cs = update(
                optimizer, true_observed, assimilated, start, end, k
            )
            lr_scheduler.step()
        cs.append(system.cs)

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


def update_last_state(
    optimizer: optim_base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
    start: int,
    end: int,
    k: int,
) -> jndarray:
    return optimizer(true_observed[end - 1], assimilated[-1])


def update_mean_state(
    optimizer: optim_base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
    start: int,
    end: int,
    k: int,
) -> jndarray:
    return optimizer(
        true_observed[start - k + 2 : end].mean(axis=0),
        assimilated[1:].mean(axis=0),
    )


def update_mean_derivative(
    optimizer: optim_base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
    start: int,
    end: int,
    k: int,
) -> jndarray:
    mean_gradient = jax.vmap(optimizer.compute_gradient, 0)(
        true_observed[start - k + 2 : end], assimilated[1:]
    ).mean(axis=0)
    step = optimizer.step_from_gradient(
        mean_gradient,
        true_observed[start - k + 2 : end].mean(axis=0),
        assimilated[1:].mean(axis=0),
    )
    return optimizer.system.cs + step
