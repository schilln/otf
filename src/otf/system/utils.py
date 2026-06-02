"""Utility functions for adapting ODEs and observation masks for use with
`BaseSystem`.

Includes helpers to flatten shaped states/ODEs and build boolean masks from
slices.
"""

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
        Callable '(params, state) -> state_dot'. The 'state' argument is
        expected to have shape 'shape'.
    shape
        Shape of the state expected by 'ode' (for example 'state0.shape').

    Returns
    -------
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        A callable '(params, flat_state) -> flat_state_dot' where both the input
        state and returned derivative are flattened 1-D arrays.
    """

    def flat_ode(ps: jndarray, state: jndarray) -> jndarray:
        return jnp.ravel(ode(ps, jnp.reshape(state, shape)))

    return flat_ode


def mask_from_slice(
    slice_obj: slice | tuple[slice | int, ...],
    shape: tuple[int, ...],
) -> jndarray:
    """Return a flattened boolean mask corresponding to 'slice_obj'.

    The returned mask is a 1-D boolean 'jnp.ndarray' of length equal to the
    product of 'shape' dimensions and can be used as an observation mask for a
    flattened state.

    Parameters
    ----------
    slice_obj
        A slice or a tuple of slices and/or integers to index an array of shape
        'shape'.
    shape
        Shape of the array that 'slice_obj' indexes (for example,
        'state0.shape').

    Returns
    -------
    jnp.ndarray
        1-D boolean mask where indexed positions are True.
    """
    mask = jnp.full(shape, False, dtype=bool)
    mask = mask.at[slice_obj].set(True)
    return jnp.ravel(mask)


def flatten_mask(
    mask: jndarray,
) -> jndarray:
    """Return a flattened boolean mask.

    Parameters
    ----------
    mask
        Boolean mask for an array of some shape.

    Returns
    -------
    jnp.ndarray
        Flattened 1-D boolean mask ('mask.ravel()').
    """
    return mask.ravel()
