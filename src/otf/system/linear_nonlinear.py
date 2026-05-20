"""Helpers for systems with a separable linear and nonlinear part.

This module provides mixins and concrete `System_*` subclasses for ODEs that can
be written as `L(c) * state + N(c, state)` where a parameter-dependent linear
operator multiplies the state and a separate nonlinear function provides the
remaining dynamics.
"""

from collections.abc import Callable

from jax import numpy as jnp

from .base import System_ModelKnown, System_ModelUnknown

jndarray = jnp.ndarray


class _LinearNonlinearMixin:
    @staticmethod
    def _define_ode(
        linear: Callable[[jndarray], jndarray],
        nonlinear_ode: Callable[[jndarray, jndarray], jndarray],
    ) -> Callable[[jndarray, jndarray], jndarray]:
        """Return an ODE combining a linear operator and a nonlinear part.

        The returned callable has signature `(cs, state) -> state_dot` and
        computes `linear(cs) * state + nonlinear_ode(cs, state)`.
        """

        def ode(cs: jndarray, state: jndarray):
            return linear(cs) * state + nonlinear_ode(cs, state)

        return ode


class _AssimilatedLinearNonlinearMixin(_LinearNonlinearMixin):
    """Mixin adding assimilated-system linear and nonlinear parts."""

    def _set_assimilated_parts(
        self,
        linear_assimilated: Callable[[jndarray], jndarray],
        nonlinear_assimilated_ode: Callable[[jndarray, jndarray], jndarray],
    ) -> None:
        """Store callables describing the assimilated system components."""

        self._linear_assimilated = linear_assimilated
        self._nonlinear_assimilated_ode = nonlinear_assimilated_ode

    linear_assimilated = property(lambda self: self._linear_assimilated)
    nonlinear_assimilated_ode = property(
        lambda self: self._nonlinear_assimilated_ode
    )

    def nudged_nonlinear_assimilated_ode(
        self,
        cs: jndarray,
        true_observed: jndarray,
        assimilated: jndarray,
    ) -> jndarray:
        """Compute the nonlinear part of assimilated dynamics with nudging.

        This applies the nonlinear assimilated ODE and then subtracts the
        nudging term on observed entries.
        """

        mask = self.observed_mask

        assimilated_p = self.nonlinear_assimilated_ode(cs, assimilated)
        assimilated_p = assimilated_p.at[mask].subtract(
            self.mu * (assimilated[mask] - true_observed)
        )

        return assimilated_p


class _TrueLinearNonlinearMixin(_LinearNonlinearMixin):
    """Mixin adding true-system linear and nonlinear parts for known models."""

    def _set_true_parts(
        self,
        linear_true: Callable[[jndarray], jndarray],
        nonlinear_true_ode: Callable[[jndarray, jndarray], jndarray],
    ) -> None:
        """Bind true-system callables to the stored true parameters `gs`.

        The mixin stores zero-argument or single-argument wrappers that evaluate
        the provided callables with `self.gs` so the resulting attributes match
        the expected signatures used elsewhere.
        """

        self._linear_true = lambda: linear_true(self.gs)
        self._nonlinear_true_ode = lambda true: nonlinear_true_ode(
            self.gs, true
        )

    linear_true = property(lambda self: self._linear_true)
    nonlinear_true_ode = property(lambda self: self._nonlinear_true_ode)


class System_LinearNonlinear_ModelKnown(
    _AssimilatedLinearNonlinearMixin,
    _TrueLinearNonlinearMixin,
    System_ModelKnown,
):
    """Concrete system: known true-model with separable linear/nonlinear parts.

    The class constructs compatible assimilated and true ODEs from provided
    linear and nonlinear component callables.
    """

    def __init__(
        self,
        mu: float,
        gs: jndarray,
        cs: jndarray,
        observed_mask: jndarray,
        linear_assimilated: Callable[[jndarray], jndarray],
        nonlinear_assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        linear_true: Callable[[jndarray], jndarray],
        nonlinear_true_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
        true_observed_mask: jndarray | None = None,
    ):
        self._set_assimilated_parts(
            linear_assimilated, nonlinear_assimilated_ode
        )
        assimilated_ode = self._define_ode(
            self.linear_assimilated, self.nonlinear_assimilated_ode
        )
        self._set_true_parts(linear_true, nonlinear_true_ode)
        true_ode = self._define_ode(linear_true, nonlinear_true_ode)
        super().__init__(
            mu,
            gs,
            cs,
            observed_mask,
            assimilated_ode,
            true_ode,
            complex_differentiation,
            true_observed_mask,
        )


class System_LinearNonlinear_ModelUnknown(
    _AssimilatedLinearNonlinearMixin,
    System_ModelUnknown,
):
    """Concrete system: unknown true-model, only assimilated parts provided."""

    def __init__(
        self,
        mu: float,
        cs: jndarray,
        observed_mask: jndarray,
        linear_assimilated: Callable[[jndarray], jndarray],
        nonlinear_assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
    ):
        self._set_assimilated_parts(
            linear_assimilated, nonlinear_assimilated_ode
        )
        assimilated_ode = self._define_ode(
            self.linear_assimilated, self.nonlinear_assimilated_ode
        )
        super().__init__(
            mu, cs, observed_mask, assimilated_ode, complex_differentiation
        )
