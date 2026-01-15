from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem
from ..parameter_update_option import UpdateOption
from .gradient_computer import GradientComputer

jndarray = jnp.ndarray


class SensitivityGradient(GradientComputer):
    def __init__(
        self,
        system: BaseSystem,
        update_option: UpdateOption = UpdateOption.last_state,
    ):
        super().__init__(system, update_option)

        match update_option:
            case UpdateOption.last_state:
                self.compute_gradient = self._last_state
            case UpdateOption.mean_state:
                self.compute_gradient = self._mean_state
            case UpdateOption.mean_gradient:
                self.compute_gradient = self._mean_derivative

    def _compute_gradient(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        diff = assimilated[self.system.observed_mask] - observed_true
        w = _compute_sensitivity(self.system, assimilated)
        m = w.shape[1]
        if self._weight is None:
            gradient = diff @ w.reshape(-1, m).conj()
        else:
            gradient = diff @ self._weight @ w.reshape(-1, m).conj()
        return gradient if self.system.cs.dtype == complex else gradient.real

    def _last_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        return self._compute_gradient(observed_true[-1], assimilated[-1])

    def _mean_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        return self._compute_gradient(
            observed_true.mean(axis=0), assimilated.mean(axis=0)
        )

    def _mean_derivative(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        return jax.vmap(self._compute_gradient, 0)(
            observed_true, assimilated
        ).mean(axis=0)


def _compute_sensitivity(system: BaseSystem, assimilated: jndarray) -> jndarray:
    """Compute the leading-order approximation of the sensitivity equations.

    Parameters
    ----------
    system
    assimilated
        Data assimilated system state

    Returns
    -------
    sensitivity
        The ith column corresponds to the asymptotic approximation of the
        ith sensitivity (i.e., w_i = dv/dc_i corresponding to the ith
        unknown parameter c_i)
    """
    return __compute_sensitivity(assimilated, system.cs, system)


@partial(jax.jit, static_argnames="system")
def __compute_sensitivity(
    assimilated: jndarray, cs: jndarray, system: BaseSystem
) -> jndarray:
    s = system
    om = s.observed_mask

    df_dc = s.df_dc(cs, assimilated)

    if s.observe_all:
        return df_dc / s.mu
    elif not s.use_unobserved_asymptotics:
        return df_dc[om] / s.mu
    else:
        df_dv_QW0 = _solve_unobserved(assimilated, cs, df_dc, system)
        return (df_dc[om] + df_dv_QW0[om]) / s.mu


@partial(jax.jit, static_argnames="system")
def _solve_unobserved(
    assimilated: jndarray, cs: jndarray, df_dc: jndarray, system: BaseSystem
) -> jndarray:
    s = system
    um = s.unobserved_mask

    df_dv = s.df_dv(cs, assimilated)

    QW0 = jnp.linalg.lstsq(df_dv[um][:, um], -df_dc[um])[0]

    return df_dv[:, um] @ QW0
