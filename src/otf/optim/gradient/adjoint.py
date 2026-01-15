from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem
from ..parameter_update_option import UpdateOption
from .gradient_computer import GradientComputer

jndarray = jnp.ndarray


class AdjointGradient(GradientComputer):
    def __init__(self, system: BaseSystem):
        super().__init__(system, UpdateOption.adjoint)

    def compute_gradient(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        adjoint = self.compute_adjoint(observed_true, assimilated)

        return _compute_gradient(
            assimilated, adjoint, self.system.df_dc, self.system.cs
        )

    def compute_adjoint(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        return _compute_adjoint(
            observed_true,
            assimilated[:, self.system.observed_mask],
            self.system.mu,
        )


@partial(jax.jit, static_argnames=("df_dc_fn",))
def _compute_gradient(
    assimilated: jndarray, adjoint: jndarray, df_dc_fn: Callable, cs: jndarray
) -> jndarray:
    df_dc = jax.vmap(df_dc_fn, (None, 0))(cs, assimilated)
    return -(jnp.expand_dims(adjoint, 1) @ df_dc).squeeze().mean(axis=0)


@jax.jit
def _compute_adjoint(
    observed_true: jndarray, observed_assimilated: jndarray, mu: float
) -> jndarray:
    return -(observed_assimilated - observed_true) / mu
