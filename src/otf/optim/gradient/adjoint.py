from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem, System_ModelUnknown
from ...time_integration.base import MultistepSolver
from .gradient_computer import GradientComputer

jndarray = jnp.ndarray


class UpdateOption(Enum):
    """Enum for selecting parameter update methods.

    Options include:

    - asymptotic: Uses the asymptotic approximation of the observed portion of
      the adjoint.
    - complete: Uses the complete adjoint via simulation.
    - unobserved: Uses the asymptotic approximation of the observed portion of
      the adjoint and simulates the unobserved portion.
    """

    asymptotic = 0
    complete = 1
    unobserved = 2


class AdjointGradient(GradientComputer):
    def __init__(
        self,
        system: BaseSystem,
        dt: float,
        update_option: UpdateOption = UpdateOption.asymptotic,
        solver: tuple[type[MultistepSolver]] | None = None,
    ):
        super().__init__(system)
        self._dt = dt

        if update_option is not UpdateOption.asymptotic:
            if solver is None:
                raise ValueError(
                    "`solver` must not be None for the given update option"
                )
            if update_option is UpdateOption.complete:
                system = CompleteSystem(system)
                self._compute_adjoint = self.compute_adjoint_complete
            elif update_option is UpdateOption.unobserved:
                system = UnobservedSystem(system)

                def compute_adjoint(observed_true, assimilated):
                    adjoint = self.compute_adjoint_asymptotic(
                        observed_true, assimilated
                    )
                    adjoint = adjoint.at[:, self.system.unobserved_mask].add(
                        self.compute_adjoint_unobserved(
                            observed_true, assimilated
                        )
                    )
                    return adjoint

                self._compute_adjoint = compute_adjoint
            else:
                raise ValueError("update option is not supported")

            if not isinstance(solver, tuple):
                solver = (solver,)
            self._solver = solver[0](system)
            for s in solver[1:]:
                self._solver = s(system, self._solver)
        else:
            self._compute_adjoint = self.compute_adjoint_asymptotic

    def compute_gradient(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        adjoint = self._compute_adjoint(observed_true, assimilated)

        return _compute_gradient(
            assimilated, adjoint, self.system.df_dc, self.system.cs
        )

    def compute_adjoint_asymptotic(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        adjoint = jnp.zeros_like(assimilated)
        adjoint = adjoint.at[:, self.system.observed_mask].set(
            _compute_adjoint_asymptotic(
                observed_true,
                assimilated[:, self.system.observed_mask],
                self.system.mu,
            )
        )
        return adjoint

    def compute_adjoint_complete(
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

    def compute_adjoint_unobserved(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        tn, n = assimilated[:, self.system.unobserved_mask].shape
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
    gradient = (
        -(jnp.expand_dims(adjoint, 1) @ df_dc).squeeze().mean(axis=0).conj()
    )
    return gradient if cs.dtype == complex else gradient.real


@jax.jit
def _compute_adjoint_asymptotic(
    observed_true: jndarray, observed_assimilated: jndarray, mu: float
) -> jndarray:
    return -(observed_assimilated - observed_true).conj() / mu


class AdjointSystem(System_ModelUnknown):
    def __init__(self, system: BaseSystem):
        super().__init__(
            None, None, system.observed_mask, system.assimilated_ode
        )
        self._system = system

        self._n = len(self.observed_mask)

    mu = property(lambda self: self._system.mu)
    cs = property(lambda self: self._system.cs)
    observed_mask = property(lambda self: self._system.observed_mask)
    df_dv_fn = property(lambda self: self._system.df_dv)


class CompleteSystem(AdjointSystem):
    def f_assimilated(
        self,
        cs: jndarray,
        assimilated__observed_diff: jndarray,
        adjoint: jndarray,
    ) -> jndarray:
        assimilated, observed_diff = (
            assimilated__observed_diff[: self._n],
            assimilated__observed_diff[self._n :].conj(),
        )
        val = self.df_dv_fn(cs, assimilated).T @ adjoint
        val = val.at[self.observed_mask].add(
            self.mu * adjoint[self.observed_mask] + observed_diff
        )
        return val


class UnobservedSystem(AdjointSystem):
    def f_assimilated(
        self,
        cs: jndarray,
        assimilated__observed_diff: jndarray,
        adjoint: jndarray,
    ) -> jndarray:
        om = self.observed_mask
        um = self.unobserved_mask

        assimilated, observed_diff = (
            assimilated__observed_diff[: self._n],
            assimilated__observed_diff[self._n :].conj(),
        )

        df_dv = self.df_dv_fn(cs, assimilated)
        val = df_dv.T[um][:, um] @ adjoint
        val = val.at[:].subtract(df_dv[um][:, om] @ observed_diff)
        return val
