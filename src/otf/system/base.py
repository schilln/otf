"""Classes to define systems of differential equations with which to use the
on-the-fly (OTF) method of data assimilation. Based on the AOT method which
"nudges" a data assimilated system toward an observed "ground truth" system,
OTF in addition estimates the model governing the observed system.
"""

from collections.abc import Callable

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

        _df_dc = jax.jacrev(
            self.assimilated_ode,
            0,
            holomorphic=self._complex_differentiation,
        )
        _df_dv = jax.jacrev(
            self.assimilated_ode,
            1,
            holomorphic=self._complex_differentiation,
        )
        if self._complex_differentiation:

            def df_dc(cs: jndarray, assimilated: jndarray) -> jndarray:
                return _df_dc(cs.astype(complex), assimilated)

            def df_dv(cs: jndarray, assimilated: jndarray) -> jndarray:
                return _df_dv(cs.astype(complex), assimilated)
        else:

            def df_dc(cs: jndarray, assimilated: jndarray) -> jndarray:
                return _df_dc(cs, assimilated)

            def df_dv(cs: jndarray, assimilated: jndarray) -> jndarray:
                return _df_dv(cs, assimilated)

        self._df_dc = df_dc
        self._df_dv = df_dv

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

    def _set_cs(self, cs):
        self._cs = cs

    # The following attributes are read-only.
    mu = property(lambda self: self._mu)
    gs = property(lambda self: self._gs)
    cs = property(lambda self: self._cs, _set_cs)
    observed_mask = property(lambda self: self._observed_mask)
    unobserved_mask = property(lambda self: self._unobserved_mask)
    observe_all = property(lambda self: self._observe_all)
    assimilated_ode = property(lambda self: self._assimilated_ode)
    df_dc = property(lambda self: self._df_dc)
    df_dv = property(lambda self: self._df_dv)
    complex_differentiation = property(
        lambda self: self._complex_differentiation
    )
    use_unobserved_asymptotics = property(
        lambda self: self._use_unobserved_asymptotics
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
        use_unobserved_asymptotics: bool = False,
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
