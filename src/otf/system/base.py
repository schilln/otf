"""Classes to define systems of differential equations with which to use the
on-the-fly (OTF) method of data assimilation. Based on the AOT method which
"nudges" a data assimilated system toward an observed "ground truth" system,
OTF in addition estimates the model governing the observed system.
"""

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

jndarray = jnp.ndarray


class BaseSystem:
    """A base class for defining systems of differential equations and to which
    the on-the-fly (OTF) method of data assimilation may be applied.
    """

    def __init__(
        self,
        mu: float,
        gs: jndarray,
        cs: jndarray,
        observed_mask: jndarray,
        assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
        use_unobserved_asymptotics: bool = False,
    ):
        """

        Parameters
        ----------
        mu
            Nudging parameter
        gs
            Parameter values to be used by the "true" system
        cs
            Estimated parameter values to be used by the data assimilate system,
            to be estimated/optimized (may or may not correspond to `gs`)
        observed_mask
            Boolean mask denoting the observed part of the data assimilated
            system states when nudging in `f_assimilated`
        assimilated_ode
            Function that computes the time derivative of the data assimilated
            state using the current estimated parameters `cs`.
            Parameters: (cs, true_observed, assimilated)
        complex_differentiation
            Set to True if state values may take complex values (e.g., if
            time integrating Fourier coefficients). This allows
            auto-differentiation to work.
        use_unobserved_asymptotics
            Set to False to not use additional asymptotic information from
            "unobserved" portion of simulated state

        Methods
        -------
        f_assimilated
            Computes the time derivative of the data assimilated state.
        compute_w
            Computes the leading-order approximation of the sensitivity
            equations.
            May be overridden (see docstring).
        """
        if not isinstance(observed_mask, jndarray):
            raise ValueError(
                "`observed_mask` must be jnp.ndarray boolean array"
            )

        self._mu = mu
        self._gs = gs
        self._observed_mask = observed_mask
        self._unobserved_mask = ~observed_mask
        self._observe_all = not jnp.any(self._unobserved_mask)
        self._cs = cs
        self._assimilated_ode = assimilated_ode

        self._complex_differentiation = complex_differentiation
        self._use_unobserved_asymptotics = use_unobserved_asymptotics

    def f_assimilated(
        self,
        cs: jndarray,
        true_observed: jndarray,
        assimilated: jndarray,
    ) -> jndarray:
        """Computes the time derivative of `assimilated` using `assimilated_ode`
        followed by nudging the data assimilated system using the observed
        portion of the the true state, `true_observed`.

        This function will be jitted.

        Parameters
        ----------
        cs
            Estimated parameter values to be used by the data assimilated system
        true_observed
            Observed portion of true system system
        assimilated
            Data assimilated system state

        Returns
        -------
        assimilated_p
            The time derivative of `assimilated_p`
        """
        mask = self.observed_mask

        assimilated_p = self._assimilated_ode(cs, assimilated)
        assimilated_p = assimilated_p.at[mask].subtract(
            self.mu * (assimilated[mask] - true_observed)
        )

        return assimilated_p

    def compute_w(self, assimilated: jndarray) -> jndarray:
        """Compute the leading-order approximation of the sensitivity equations.

        Subclasses may override this method to optimize computation or to obtain
        higher-order approximations.

        Parameters
        ----------
        assimilated
            Data assimilated system state

        Returns
        -------
        W
            The ith column corresponds to the asymptotic approximation of the
            ith sensitivity (i.e., w_i = dv/dc_i corresponding to the ith
            unknown parameter c_i)
        """
        return self._compute_w(self.cs, assimilated)

    @partial(jax.jit, static_argnames="self")
    def _compute_w(self, cs: jndarray, assimilated: jndarray) -> jndarray:
        om = self.observed_mask

        df_dc = jax.jacrev(
            self._assimilated_ode,
            0,
            holomorphic=self.complex_differentiation,
        )(cs, assimilated)

        if self._observe_all:
            return df_dc / self.mu
        elif not self._use_unobserved_asymptotics:
            return df_dc[om] / self.mu
        else:
            df_dv_QW0 = self._solve_unobserved(cs, assimilated, df_dc)
            return (df_dc[om] + df_dv_QW0[om]) / self.mu

    @partial(jax.jit, static_argnames="self")
    def _solve_unobserved(
        self, cs: jndarray, assimilated: jndarray, df_dc: jndarray
    ) -> jndarray:
        um = self.unobserved_mask

        df_dv = jax.jacrev(
            self._assimilated_ode,
            1,
            holomorphic=self.complex_differentiation,
        )(cs, assimilated)

        QW0 = jnp.linalg.lstsq(df_dv[um][:, um], -df_dc[um])[0]

        return df_dv[:, um] @ QW0

    def _set_cs(self, cs):
        self._cs = cs

    # The following attributes are read-only.
    mu = property(lambda self: self._mu)
    gs = property(lambda self: self._gs)
    cs = property(lambda self: self._cs, _set_cs)
    observed_mask = property(lambda self: self._observed_mask)
    unobserved_mask = property(lambda self: self._unobserved_mask)
    complex_differentiation = property(
        lambda self: self._complex_differentiation
    )


class System_ModelKnown(BaseSystem):
    """Class for when the true differential equation is known and may be
    simulated concurrently with the data assimilated system.

    See `BaseSystem` for additional documentation.

    Methods
    -------
    f_true
        Computes the time derivative of the true system, given the current
        state.
    """

    def __init__(
        self,
        mu: float,
        gs: jndarray,
        cs: jndarray,
        observed_mask: jndarray,
        assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        true_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
        use_unobserved_asymptotics: bool = True,
        true_observed_mask: jndarray | None = None,
    ):
        """

        Parameters
        ----------
        See `BaseSystem` for other parameter definitions.

        true_ode
            Function that computes the time derivative of the true state
            Parameters: (gs, true)
        true_observed_mask
            Boolean mask denoting the observed part of the true system states.
            If None or not provided, assumed equal to `observed_mask`.
        """
        super().__init__(
            mu,
            gs,
            cs,
            observed_mask,
            assimilated_ode,
            complex_differentiation,
            use_unobserved_asymptotics,
        )

        self._true_ode = true_ode
        self._true_observed_mask = (
            true_observed_mask
            if true_observed_mask is not None
            else observed_mask
        )

    def f_true(
        self,
        true: jndarray,
    ) -> jndarray:
        """Computes the time derivative of `true` using `true_ode`.

        This function will be jitted.

        Parameters
        ----------
        true
            True system state

        Returns
        -------
        true_p
            The time derivative of `assimilated_p`
        """
        return self._true_ode(self.gs, true)

    true_observed_mask = property(lambda self: self._true_observed_mask)


class System_ModelUnknown(BaseSystem):
    """Class for when the true differential equation is not known and and so may
    not be simulated concurrently with the data assimilated system.

    See `BaseSystem` for additional documentation.
    """
