from collections.abc import Callable

from jax import numpy as jnp

from . import BaseSystem, System_ModelKnown, System_ModelUnknown

jndarray = jnp.ndarray


class System_LinearNonlinear(BaseSystem):
    def __init__(
        self,
        mu: float,
        cs: jndarray,
        observed_mask: jndarray,
        linear_assimilated: Callable[[jndarray, jndarray], jndarray],
        nonlinear_assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
    ):
        assimilated_ode = self._define_ode(
            linear_assimilated, nonlinear_assimilated_ode
        )
        BaseSystem.__init__(
            mu, cs, observed_mask, assimilated_ode, complex_differentiation
        )

        self._linear_assimilated = linear_assimilated
        self._nonlinear_assimilated_ode = nonlinear_assimilated_ode

    @staticmethod
    def _define_ode(
        linear: Callable[[jndarray], jndarray],
        nonlinear_ode: Callable[[jndarray, jndarray], jndarray],
    ) -> Callable[[jndarray, jndarray], jndarray]:
        def ode(cs: jndarray, state: jndarray):
            return linear(cs) @ state + nonlinear_ode(cs, state)

        return ode


class System_LinearNonlinear_ModelKnown(
    System_LinearNonlinear, System_ModelKnown
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
        System_LinearNonlinear().__init__(
            mu,
            cs,
            observed_mask,
            linear_assimilated,
            nonlinear_assimilated_ode,
            complex_differentiation,
        )

        assimilated_ode = self._assimilated_ode
        true_ode = self._define_ode(linear_true, nonlinear_true_ode)

        System_ModelKnown().__init__(
            mu,
            gs,
            cs,
            observed_mask,
            assimilated_ode,
            true_ode,
            complex_differentiation,
            true_observed_mask,
        )

        self._linear_true = linear_true
        self._nonlinear_true_ode = nonlinear_true_ode


class System_LinearNonlinear_ModelUnknown(
    System_LinearNonlinear, System_ModelUnknown
):
    def __init__(self, *args, **kwargs):
        System_LinearNonlinear(*args, **kwargs)
