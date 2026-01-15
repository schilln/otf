from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem

jndarray = jnp.ndarray


def compute_sensitivity(system: BaseSystem, assimilated: jndarray) -> jndarray:
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
    return _compute_sensitivity(assimilated, system.cs, system)


@partial(jax.jit, static_argnames="system")
def _compute_sensitivity(
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
