from collections.abc import Callable

from jax import numpy as jnp

from . import System_ModelKnown, System_ModelUnknown

jndarray = jnp.ndarray


class _LinearNonlinearMixin:
    @staticmethod
    def _define_ode(
        linear: Callable[[jndarray], jndarray],
        nonlinear_ode: Callable[[jndarray, jndarray], jndarray],
    ) -> Callable[[jndarray, jndarray], jndarray]:
        def ode(cs: jndarray, state: jndarray):
            return linear(cs) @ state + nonlinear_ode(cs, state)

        return ode


class _AssimilatedLinearNonlinearMixin(_LinearNonlinearMixin):
    def _set_assimilated_parts(
        self,
        linear_assimilated: Callable[[jndarray], jndarray],
        nonlinear_assimilated_ode: Callable[[jndarray, jndarray], jndarray],
    ) -> None:
        self._linear_assimilated = linear_assimilated
        self._nonlinear_assimilated_ode = nonlinear_assimilated_ode

    linear_assimilated = property(lambda self: self._linear_assimilated)
    nonlinear_assimilated_ode = property(
        lambda self: self._nonlinear_assimilated_ode
    )


class _TrueLinearNonlinearMixin(_LinearNonlinearMixin):
    def _set_true_parts(
        self,
        linear_true: Callable[[jndarray], jndarray],
        nonlinear_true_ode: Callable[[jndarray, jndarray], jndarray],
    ) -> None:
        self._linear_true = linear_true
        self._nonlinear_true_ode = nonlinear_true_ode

    linear_true = property(lambda self: self._linear_true)
    nonlinear_true_ode = property(lambda self: self._nonlinear_true_ode)


class System_LinearNonlinear_ModelKnown(
    System_ModelKnown,
    _AssimilatedLinearNonlinearMixin,
    _TrueLinearNonlinearMixin,
):
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
    System_ModelUnknown,
    _AssimilatedLinearNonlinearMixin,
):
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
