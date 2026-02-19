from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem, System_ModelUnknown
from ...time_integration.base import MultistepSolver, SinglestepSolver
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
        update_option: UpdateOption = UpdateOption.asymptotic,
        solver: tuple[type[SinglestepSolver | MultistepSolver]]
        | type[SinglestepSolver | MultistepSolver]
        | None = None,
        dt: float | None = None,
        interval_fraction: float = 1,
    ):
        super().__init__(system)

        if not (0 < interval_fraction <= 1):
            raise ValueError(
                "`interval_fraction` should be in (0, 1]"
                f" (was {interval_fraction})"
            )
        self._interval_fraction = interval_fraction

        self._compute_adjoint = self._set_up_adjoint_method(update_option)

        if update_option is not UpdateOption.asymptotic:
            if dt is None:
                raise ValueError("`dt` must not be None for this update option")
            if solver is None:
                raise ValueError(
                    "`solver` must not be None for the given update option"
                )

            adjoint_system = self._set_up_adjoint_system(system, update_option)

            self._dt = dt
            self._solver = self._set_up_solver(adjoint_system, solver)

    def compute_gradient(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        frac = self._interval_fraction
        n = len(observed_true)
        s_ = jnp.s_[-round(n * frac) :]
        adjoint = self._compute_adjoint(observed_true[s_], assimilated[s_])

        return self._compute_gradient(
            assimilated[s_], adjoint, self.system.df_dc, self.system.cs
        )

    # Initialization

    def _set_up_adjoint_method(self, update_option: UpdateOption) -> Callable:
        """Select the method to compute the adjoint."""
        match update_option:
            case UpdateOption.asymptotic:
                return self._compute_adjoint_asymptotic
            case UpdateOption.complete:
                return self._compute_adjoint_complete
            case UpdateOption.unobserved:
                return self._compute_adjoint_unobserved
            case _:
                raise ValueError("update option is not supported")

    def _set_up_adjoint_system(
        self,
        system: BaseSystem,
        update_option: UpdateOption,
    ) -> AdjointSystem:
        """Create the adjoint system."""
        if update_option is UpdateOption.complete:
            return CompleteSystem(system)
        elif update_option is UpdateOption.unobserved:
            return UnobservedSystem(system)
        else:
            raise ValueError("update option is not supported")

    def _set_up_solver(
        self,
        system: BaseSystem,
        solver: tuple[type[SinglestepSolver | MultistepSolver]]
        | type[SinglestepSolver | MultistepSolver],
    ) -> SinglestepSolver | MultistepSolver:
        """Initialize and chain solvers for the given system."""
        if not isinstance(solver, tuple):
            solver = (solver,)

        _solver = solver[0](system)
        for s in solver[1:]:
            _solver = s(system, _solver)

        return _solver

    # Adjoint computation

    @partial(jax.jit, static_argnames=("self",))
    def _compute_adjoint_asymptotic(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        adjoint = jnp.zeros_like(assimilated)
        adjoint = adjoint.at[:, self.system.observed_mask].set(
            -(assimilated[:, self.system.observed_mask] - observed_true).conj()
            / self.system.mu
        )
        return adjoint

    def _compute_adjoint_complete(
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
            adjoint0, tf, self._dt, -self._dt, assimilated__observed_diff[::-1]
        )
        return adjoint[::-1]

    def _compute_adjoint_unobserved(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        adjoint = self._compute_adjoint_asymptotic(observed_true, assimilated)
        adjoint = adjoint.at[:, self.system.unobserved_mask].add(
            self._compute_adjoint_unobserved_only(observed_true, assimilated)
        )
        return adjoint

    def _compute_adjoint_unobserved_only(
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
            adjoint0, tf, self._dt, -self._dt, assimilated__observed_diff[::-1]
        )
        return adjoint[::-1] / self.system.mu

    @staticmethod
    @partial(jax.jit, static_argnames=("df_dc_fn",))
    def _compute_gradient(
        assimilated: jndarray,
        adjoint: jndarray,
        df_dc_fn: Callable,
        cs: jndarray,
    ) -> jndarray:
        df_dc = jax.vmap(df_dc_fn, (None, 0))(cs, assimilated)
        gradient = (
            -(jnp.expand_dims(adjoint, 1) @ df_dc).squeeze().mean(axis=0).conj()
        )
        return gradient if cs.dtype == complex else gradient.real


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
        df_dv = self.df_dv_fn(cs, assimilated)
        val = adjoint @ df_dv
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
        val = adjoint @ df_dv[um][:, um]
        val = val.at[:].subtract(observed_diff @ df_dv[om][:, um])
        return val
