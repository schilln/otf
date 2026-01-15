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
        df_dc = jax.vmap(self.system.df_dc, (None, 0))(
            self.system.cs, assimilated
        )

        return -(jnp.expand_dims(adjoint, 1) @ df_dc).squeeze().mean(axis=0)

    def compute_adjoint(self, observed_true: jndarray, assimilated: jndarray):
        om = self.system.observed_mask

        return -(assimilated[:, om] - observed_true) / self.system.mu
