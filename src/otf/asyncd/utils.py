"""
`run_update` iteratively simulates a `System` and updates parameter values,
returning the sequences of parameter values and data assimilated-vs-true errors.
"""

from collections.abc import Callable
from enum import Enum

import jax
import numpy as np
import scipy
from jax import lax
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
    direct_simulation = 3


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

    if return_all:
        assimilateds = [np.expand_dims(assimilated0, 0)]

    match parameter_update_option:
        case ParameterUpdateOption.last_state:
            update = update_last_state
        case ParameterUpdateOption.mean_state:
            update = update_mean_state
        case ParameterUpdateOption.mean_gradient:
            update = update_mean_derivative
        case ParameterUpdateOption.direct_simulation:
            system._dt = dt
            update = update_direct_sim

    if weight is None:
        norm = np.linalg.norm
    if weight is not None:
        sqrt_weight = scipy.linalg.sqrtm(weight)
        norm = lambda states, *args, **kwargs: np.linalg.norm(
            sqrt_weight @ states.T, *args, **kwargs
        )

    t0 = T0
    tf = t0 + t_relax

    num_steps = assimilated_solver.compute_num_steps(t0, tf, dt)
    end = num_steps
    assimilated, tls = assimilated_solver.solve_assimilated(
        assimilated0, t0, tf, dt, true_observed[:end]
    )

    if return_all:
        assimilateds.append(assimilated[1:])

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
        norm(true_compare[1:end] - assimilated_compare(assimilated[1:]))
        / norm(true_compare[1:end])
    )

    start = end - 1

    while tf <= Tf:
        num_steps = assimilated_solver.compute_num_steps(t0, tf, dt)
        end += num_steps - (k - 1)
        assimilated, tls = assimilated_solver.solve_assimilated(
            assimilated0,
            t0,
            tf,
            dt,
            true_observed[start - k + 1 : end],
            **assimilated_args,
        )

        if return_all:
            assimilateds.append(assimilated[assimilated_solver.k :])

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
            norm(
                true_compare[start + 1 : end]
                - assimilated_compare(assimilated[k:])
            )
            / norm(true_compare[start + 1 : end])
        )

        start = end - 1

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
    return optimizer.system.cs + jnp.real(step)


def update_direct_sim(
    optimizer: optim_base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
    start: int,
    end: int,
    k: int,
) -> jndarray:
    t, a = true_observed[start - k + 2 : end], assimilated[1:]
    system = optimizer.system
    cs = system.cs
    om, um = system.observed_mask, system.unobserved_mask

    df_dvs = jax.vmap(system._df_dv, (None, 0))(cs, assimilated)
    df_dcs = jax.vmap(system._df_dc, (None, 0))(cs, assimilated)

    x0 = jnp.zeros_like(df_dcs[0][um])
    dt = system._dt
    n = assimilated.shape[0]
    qw0 = _solve(x0, df_dvs[:, um][:, :, um], df_dcs[:, um], dt)
    # qw0 = jnp.linalg.lstsq(df_dv[:, um][:, :, um], -df_dc)

    w = df_dcs[1:, om] - (df_dvs[:, :, um] @ qw0)[1:, om]
    w /= system.mu

    diff = a[:, om] - t
    m = w.shape[2]
    gradient = jnp.real(
        jnp.expand_dims(diff.conj(), 1) @ w.reshape(n - 1, -1, m)
    ).squeeze(1)
    step = optimizer.step_from_gradient(
        gradient.mean(axis=0), t.mean(axis=0), a.mean(axis=0)
    )
    return optimizer.system.cs + jnp.real(step)


def _solve(x0, df_dvs, df_dcs, dt):
    n = df_dvs.shape[0]
    xs = jnp.full((n, *x0.shape), jnp.inf)
    xs = xs.at[0].set(x0)

    (xs,), _ = lax.fori_loop(1, n, _loop, ((xs,), (df_dvs, df_dcs, dt)))

    return xs


def _loop(i, vals):
    ((xs,), (df_dvs, df_dcs, dt)) = vals
    df_dv = df_dvs[i]
    df_dc = df_dcs[i]
    xs = xs.at[i].set(xs[i - 1] + dt * f(xs[i - 1], df_dv, df_dc))
    return (xs,), (df_dvs, df_dcs, dt)


def f(x, df_dv, df_dc):
    return -df_dv @ x - df_dc
