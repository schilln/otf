from collections.abc import Callable
from enum import Enum

import jax
from jax import numpy as jnp

from ..optim import base

jndarray = jnp.ndarray


class ParameterUpdateOption(Enum):
    """Enum for selecting parameter update methods.

    Options include:

    - last_state: Uses the last observed state for the update.
    - mean_state: Uses the mean of the observed states for the update.
    - mean_gradient: Uses the mean of the gradients for the update.
    """

    last_state = 0
    mean_state = 1
    mean_gradient = 2


def get_update_function(
    parameter_update_option: ParameterUpdateOption,
) -> Callable[
    [base.BaseOptimizer, jndarray, jndarray, int, int, int], jndarray
]:
    """Returns a function to update parameters.

    Parameters
    ----------
    parameter_update_option
        Enum indicating which method of parameter update to use

    Returns
    -------
    update
        Function that takes

        - an optimizer,
        - true observed values,
        - data assimilated values,
        - start index for `true_observed` (inclusive),
        - end index for `true_observed` (exclusive), and
        - the number of steps used in the time integration method (e.g., a
          single-step solver like forward Euler uses 1 step)

        and returns the updated parameter values.

    """
    match parameter_update_option:
        case ParameterUpdateOption.last_state:
            update = _update_last_state
        case ParameterUpdateOption.mean_state:
            update = _update_mean_state
        case ParameterUpdateOption.mean_gradient:
            update = _update_mean_derivative

    return update


def _update_last_state(
    optimizer: base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
    start: int,
    end: int,
    num_multistep: int,
) -> jndarray:
    return optimizer(true_observed[end - 1], assimilated[-1])


def _update_mean_state(
    optimizer: base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
    start: int,
    end: int,
    num_multistep: int,
) -> jndarray:
    return optimizer(
        true_observed[start - num_multistep + 2 : end].mean(axis=0),
        assimilated[1:].mean(axis=0),
    )


def _update_mean_derivative(
    optimizer: base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
    start: int,
    end: int,
    num_multistep: int,
) -> jndarray:
    mean_gradient = jax.vmap(optimizer.compute_gradient, 0)(
        true_observed[start - num_multistep + 2 : end], assimilated[1:]
    ).mean(axis=0)
    step = optimizer.step_from_gradient(
        mean_gradient,
        true_observed[start - num_multistep + 2 : end].mean(axis=0),
        assimilated[1:].mean(axis=0),
    )
    return optimizer.system.cs + step
