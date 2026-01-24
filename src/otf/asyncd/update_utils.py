from collections.abc import Callable

from jax import numpy as jnp

from .. import optim
from ..optim import base
from ..optim.gradient import sensitivity

jndarray = jnp.ndarray


def get_update_function(
    optimizer: base.BaseOptimizer,
) -> Callable[
    [base.BaseOptimizer, jndarray, jndarray, int, int, int], jndarray
]:
    """Returns a function to update parameters.

    Parameters
    ----------
    optimizer

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
    if isinstance(optimizer.gradient_computer, optim.SensitivityGradient):
        match optimizer.gradient_computer.update_option:
            case sensitivity.UpdateOption.last_state:
                update = _last_state
            case (
                sensitivity.UpdateOption.mean_state
                | sensitivity.UpdateOption.mean_gradient
            ):
                update = _multiple_state
            case _:
                raise NotImplementedError("update option is not supported")
    elif isinstance(optimizer.gradient_computer, optim.AdjointGradient):
        update = _multiple_state
    else:
        raise NotImplementedError("gradient computer is not supported")

    return update


def _last_state(
    optimizer: base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
) -> jndarray:
    # Taking a slice of one item instead of just indexing (e.g., array[-1])
    # preserves the number of dimension of the resulting array (i.e., 2D).
    return optimizer(true_observed[-1:], assimilated[-1:])


def _multiple_state(
    optimizer: base.BaseOptimizer,
    true_observed: jndarray,
    assimilated: jndarray,
) -> jndarray:
    return optimizer(true_observed, assimilated)
