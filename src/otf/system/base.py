"""Classes to define systems of differential equations with which to use the
on-the-fly (OTF) method of data assimilation. Based on the AOT method which
"nudges" a data assimilated system toward an observed "ground truth" system, OTF
in addition estimates the model governing the observed system.
"""

from collections.abc import Callable

import jax
from jax import numpy as jnp

jndarray = jnp.ndarray


class BaseSystem:
    """Base abstraction for a dynamical system used with OTF assimilation.

    This class wraps an ODE for an assimilated system together with nudging
    behavior that pushes the assimilated state toward observed portions of a
    (possibly partially observed) true state. Subclasses may provide a known
    true system or leave the true system unspecified.
    """

    def __init__(
        self,
        mu: float,
        cs: jndarray,
        observed_mask: jndarray,
        assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
    ):
        """Initialize the base system.

        Parameters
        ----------
        mu
            Nudging parameter.
        cs
            Estimated parameter values used by the assimilated system (may
            differ from the true system parameters `gs` used by subclasses).
        observed_mask
            Boolean `jnp.ndarray` mask indicating observed entries of a
            flattened state. Nudging is applied only on these entries.
        assimilated_ode
            Callable `(cs, state) -> state_dot` producing the time derivative
            for the assimilated system given parameters `cs`.
        complex_differentiation
            If True, treat arrays as potentially complex for autodiff.
        """
        if not isinstance(observed_mask, jndarray):
            raise ValueError(
                "`observed_mask` must be jnp.ndarray boolean array"
            )

        self._mu = mu
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

    def f_assimilated(
        self,
        cs: jndarray,
        true_observed: jndarray,
        assimilated: jndarray,
    ) -> jndarray:
        """Return time derivative of the assimilated state with nudging.

        The method applies `assimilated_ode` and then subtracts a nudging term
        on observed entries: `mu * (assimilated - true_observed)`.

        This method is suitable for JIT compilation.

        Parameters
        ----------
        cs
            Estimated parameter values for the assimilated ODE.
        true_observed
            Observed portion of the true state (matches `observed_mask`).
        assimilated
            Current assimilated (flattened) state.

        Returns
        -------
        jnp.ndarray
            Time derivative of `assimilated` after applying nudging.
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


class System_ModelKnown(BaseSystem):
    """System where the true ODE is known and can be simulated.

    This subclass stores `gs` (true-system parameters) and a `true_ode` allowing
    simultaneous integration of the true and assimilated systems.
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
        true_observed_mask: jndarray | None = None,
    ):
        """Initialize a System_ModelKnown with a provided true ODE.

        Parameters
        ----------
        See `BaseSystem` for other parameter definitions.

        gs
            True-system parameter values used by `true_ode`.
        true_ode
            Callable `(gs, true_state) -> true_state_dot` describing the
            dynamics of the true system.
        true_observed_mask
            Boolean mask indicating observed entries of the true state. If
            `None`, the value of `observed_mask` is reused.
        """
        super().__init__(
            mu, cs, observed_mask, assimilated_ode, complex_differentiation
        )

        self._gs = gs
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
        """Return the time derivative of the true state using `true_ode`.

        This method is suitable for JIT compilation.

        Parameters
        ----------
        true
            Current true (flattened) state.

        Returns
        -------
        jnp.ndarray
            Time derivative of `true`.
        """
        return self._true_ode(self.gs, true)

    gs = property(lambda self: self._gs)
    true_observed_mask = property(lambda self: self._true_observed_mask)


class System_ModelUnknown(BaseSystem):
    """System where the true ODE is unknown and cannot be simulated.

    This subclass does not provide a `true_ode` or `gs`; it is suitable when
    only an assimilated model is available and the true dynamics cannot be
    integrated alongside the assimilated system.

    See `BaseSystem` for shared behavior and API.
    """
