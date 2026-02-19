from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem
from .gradient_computer import GradientComputer

jndarray = jnp.ndarray


class UpdateOption(Enum):
    """Enum for selecting parameter update methods.

    Options include:

    - last_state: Uses the last observed state for the update.
    - mean_state: Uses the mean of the observed states for the update.
    - mean_gradient: Uses the mean of the gradients for the update.
    """

    last_state = 0
    mean_state = 1
    mean_gradient = 2


class SensitivityGradient(GradientComputer):
    def __init__(
        self,
        system: BaseSystem,
        update_option: UpdateOption = UpdateOption.last_state,
    ):
        super().__init__(system)

        self._update_option = update_option
        self.compute_gradient = self._set_up_gradient(update_option)

    def _set_up_gradient(self, update_option: UpdateOption) -> Callable:
        match update_option:
            case UpdateOption.last_state:
                return self._last_state
            case UpdateOption.mean_state:
                return self._mean_state
            case UpdateOption.mean_gradient:
                return self._mean_derivative
            case _:
                raise NotImplementedError("update option is not supported")

    def _last_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        sensitivity = self._compute_sensitivity_asymptotic(
            self.system, assimilated[-1:], self.system.cs
        )

        return self._compute_gradient(
            observed_true[-1:],
            assimilated[-1:],
            self.system.cs,
            sensitivity,
        ).squeeze(axis=0)

    def _mean_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        assimilated_mean = assimilated.mean(axis=0, keepdims=True)
        sensitivity = self._compute_sensitivity_asymptotic(
            self.system, assimilated_mean, self.system.cs
        )

        return self._compute_gradient(
            observed_true.mean(axis=0),
            assimilated_mean,
            self.system.cs,
            sensitivity,
        ).squeeze(axis=0)

    def _mean_derivative(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        sensitivity = self._compute_sensitivity_asymptotic(
            self.system, assimilated, self.system.cs
        )

        return self._compute_gradient(
            observed_true, assimilated, self.system.cs, sensitivity
        ).mean(axis=0)

    @partial(jax.jit, static_argnames=("self",))
    def _compute_gradient(
        self,
        observed_true: jndarray,
        assimilated: jndarray,
        cs: jndarray,
        sensitivity: jndarray,
    ) -> jndarray:
        diff = assimilated[:, self.system.observed_mask] - observed_true
        if self._weight is None:
            gradient = (jnp.expand_dims(diff, 1) @ sensitivity.conj()).squeeze(
                axis=1
            )
        else:
            gradient = (
                jnp.expand_dims(diff, 1) @ self._weight @ sensitivity.conj()
            ).squeeze(axis=1)
        return gradient if cs.dtype == complex else gradient.real

    @staticmethod
    @partial(jax.jit, static_argnames="system")
    def _compute_sensitivity_asymptotic(
        system, assimilated: jndarray, cs: jndarray
    ) -> jndarray:
        """Compute the leading-order approximation of the sensitivity equations.

        Parameters
        ----------
        system
        assimilated
            Data assimilated system state
        cs
            System parameters

        Returns
        -------
        sensitivity
            The ith column corresponds to the asymptotic approximation of the
            ith sensitivity (i.e., w_i = dv/dc_i corresponding to the ith
            unknown parameter c_i)
        """
        s = system
        om = s.observed_mask

        df_dc = jax.vmap(s.df_dc, (None, 0))(cs, assimilated)

        if s.observe_all:
            return df_dc / s.mu
        elif not s.use_unobserved_asymptotics:
            return df_dc[:, om] / s.mu
        else:
            df_dv_QW0 = _solve_unobserved(assimilated, cs, df_dc, s)
            return (df_dc[:, om] + df_dv_QW0[:, om]) / s.mu

    update_option = property(lambda self: self._update_option)


@partial(jax.jit, static_argnames="system")
def _solve_unobserved(
    assimilated: jndarray, cs: jndarray, df_dc: jndarray, system: BaseSystem
) -> jndarray:
    s = system
    um = s.unobserved_mask

    df_dv = jax.vmap(s.df_dv, (None, 0))(cs, assimilated)

    QW0 = jax.vmap(jnp.linalg.lstsq)(df_dv[:, um][:, :, um], -df_dc[:, um])[0]

    return df_dv[:, :, um] @ QW0
