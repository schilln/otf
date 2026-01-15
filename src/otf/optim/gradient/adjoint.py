from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem, System_ModelUnknown
from ..parameter_update_option import UpdateOption
from .gradient_computer import GradientComputer
from ...time_integration.base import MultistepSolver

jndarray = jnp.ndarray


class AdjointGradient(GradientComputer):
    def __init__(
        self,
        system: BaseSystem,
        solver: tuple[type[MultistepSolver]],
        dt: float,
    ):
        super().__init__(system, UpdateOption.adjoint)

        self._adjoint_system = AdjointSystem(system)
        if not isinstance(solver, tuple):
            solver = (solver,)
        self._solver = solver[0](self._adjoint_system)
        for s in solver[1:]:
            self._solver = s(self._adjoint_system, self._solver)

        self._dt = dt

    def compute_gradient(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        adjoint = self.compute_adjoint_sim(observed_true, assimilated)

        return _compute_gradient(
            assimilated, adjoint, self.system.df_dc, self.system.cs
        )

    def compute_adjoint(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        adjoint = jnp.zeros_like(assimilated)
        adjoint = adjoint.at[:, self.system.observed_mask].set(
            _compute_adjoint(
                observed_true,
                assimilated[:, self.system.observed_mask],
                self.system.mu,
            )
        )
        return adjoint

    def compute_adjoint_sim(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        tn, n = assimilated.shape
        tf = self._dt * tn

        adjoint0 = jnp.zeros(n, dtype=assimilated.dtype)

        observed_diff = (
            assimilated[:, self.system.observed_mask] - observed_true
        )
        assimilated__observed_diff = jnp.concat(
            (assimilated, observed_diff), axis=1
        )

        adjoint, _ = self._solver.solve_assimilated(
            adjoint0, tf, self._dt, -self._dt, assimilated__observed_diff
        )
        return adjoint


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


class AdjointSystem(System_ModelUnknown):
    def __init__(self, system: BaseSystem):
        super().__init__(
            None, None, None, system.observed_mask, system.assimilated_ode
        )
        self._system = system

        self._n = len(self.observed_mask)

    mu = property(lambda self: self._system.mu)
    cs = property(lambda self: self._system.cs)
    observed_mask = property(lambda self: self._system.observed_mask)
    df_dv_fn = property(lambda self: self._system.df_dv)

    def f_assimilated(
        self,
        cs: jndarray,
        assimilated__observed_diff: jndarray,
        adjoint: jndarray,
    ) -> jndarray:
        assimilated, observed_diff = (
            assimilated__observed_diff[: self._n],
            assimilated__observed_diff[self._n :],
        )
        val = self.df_dv_fn(cs, assimilated).T @ adjoint
        val = val.at[self.observed_mask].add(
            self.mu * adjoint[self.observed_mask] + observed_diff
        )
        return val
