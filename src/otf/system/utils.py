from collections.abc import Callable

from jax import numpy as jnp

jndarray = jnp.ndarray


def flatten_ode(
    ode: Callable[[jndarray, jndarray], jndarray],
    shape: tuple[int, ...],
) -> Callable[[jndarray, jndarray], jndarray]:
    """Return a flattened version of `ode` for use with `BaseSystem`.

    Parameters
    ----------
    ode
        Callable(params, state) -> time derivative of state. The `state` the
        callable expects has shape `shape`.
    shape
        Shape of the state expected by `ode` (for example, state0.shape).

    Returns
    -------
    flat_ode
        Callable(params, flat_state) -> flat time derivative of state.
    """

    def flat_ode(ps: jndarray, state: jndarray) -> jndarray:
        return jnp.ravel(ode(ps, jnp.reshape(state, shape)))

    return flat_ode


def mask_from_slice(
    slice_obj: slice | tuple[slice | int, ...],
    shape: tuple[int, ...],
) -> jndarray:
    """Return a flat boolean mask corresponding to `slice_obj`.

    For example, the mask can be used as an observation mask over a flattened
    state.

    Parameters
    ----------
    slice_obj
        A slice or a tuple of slices and/or integers to index a state of the
        given shape
    shape
        Shape of the state that `slice_obj` indexes (for example, state0.shape).

    Returns
    -------
    mask
        1D boolean mask of length prod(shape)
    """
    mask = jnp.full(shape, False, dtype=bool)
    mask = mask.at[slice_obj].set(True)
    return jnp.ravel(mask)


def flatten_mask(
    mask: jndarray,
) -> jndarray:
    """Return a flattened boolean mask.

    For example, the mask can be used as an observation mask over a flattened
    state.

    Parameters
    ----------
    mask
        Boolean mask for a state of the given shape

    Returns
    -------
    flat_mask
        1D boolean mask (mask.ravel())
    """
    return mask.ravel()
