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
        bs: jndarray,
        cs: jndarray,
        observed_slice: slice,
        assimilated_ode: Callable[[jndarray, jndarray], jndarray],
    ):
        """

        Parameters
        ----------
        mu
            Nudging parameter
        gs
            Parameter values to be used by the "true" system
        bs
            Fixed parameter values to be used by the nudged system but not to be
            updated (i.e., not to be estimated nor optimized)
        cs
            Estimated parameter values to be used by the nudged system, to be
            estimated/optimized (may or may not correspond to `gs`)
        observed_slice
            The slice denoting the observed part of the true and nudged system
            states when nudging in `f`. May use `jnp.s_` to define slice to use.
            To observed the entire system, use `jnp.s_[:]`.
        assimilated_ode
            Function that computes the time derivative of the data assimilated
            state using the current estimated parameters `cs`.
            Parameters: (cs, true_observed, assimilated)

        Methods
        -------
        assimilated_f
            Computes the time derivative of the data assimilated state.
        compute_w
            Computes the leading-order approximation of the sensitivity
            equations.
            May be overridden (see docstring).
        """
        self._mu = mu
        self._gs = gs
        self._bs = bs
        self._observed_slice = observed_slice
        self._cs = cs
        self._assimilated_ode = assimilated_ode

    def assimilated_f(
        self,
        cs: jndarray,
        true_observed: jndarray,
        nudged: jndarray,
    ) -> jndarray:
        """Computes the time derivative of `nudged` using
        `System.assimilated_ode` followed by nudging the nudged system using the
        observed portion of the the true state, `true_observed`.

        This function will be jitted.

        Parameters
        ----------
        cs
            Estimated parameter values to be used by the nudged system
        true_observed
            Observed portion of true system system
        nudged
            Nudged system state

        Returns
        -------
        nudgedp
            The time derivatives of `true` and `nudged`
        """
        s = self.observed_slice

        nudgedp = self._assimilated_ode(cs, nudged)
        nudgedp = nudgedp.at[s].subtract(self.μ * (nudged[s] - true_observed))

        return nudgedp

    def compute_w(self, nudged: jndarray) -> jndarray:
        """Compute the leading-order approximation of the sensitivity equations.

        Subclasses may override this method to optimize computation or to obtain
        higher-order approximations.

        Parameters
        ----------
        nudged
            The nudged system

        Returns
        -------
        W
            The ith row corresponds to the asymptotic approximation of the ith
            sensitivity corresponding to the ith unknown parameter ci
        """
        return self._compute_w(self.cs, nudged)

    @partial(jax.jit, static_argnames="self")
    def _compute_w(self, cs: jndarray, nudged: jndarray) -> jndarray:
        return (
            jax.jacrev(self.estimated_ode, 0)(cs, nudged)[self.observed_slice].T
            / self.μ
        )

    def _set_cs(self, cs):
        self._cs = cs

    # The following attributes are read-only.
    μ = property(lambda self: self._μ)
    gs = property(lambda self: self._gs)
    bs = property(lambda self: self._bs)
    cs = property(lambda self: self._cs, _set_cs)
    observed_slice = property(lambda self: self._observed_slice)


class SyncSystem(BaseSystem):
    """Class for when the true differential equation is known and may be
    simulated concurrently with the data assimilated system.

    See `BaseSystem` for additional documentation.

    Methods
    -------
    f_true
        Computes the time derivative of the true system, given the current
        states.
    """

    def __init__(
        self,
        mu: float,
        gs: jndarray,
        bs: jndarray,
        cs: jndarray,
        observed_slice: slice,
        assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        true_ode: Callable[[jndarray], jndarray],
    ):
        """

        Parameters
        ----------
        See `BaseSystem` for other parameter definitions.

        true_ode
            Function computing the time derivative of the true state.
            Parameters: (true,)
        """
        super().__init__(mu, gs, bs, cs, observed_slice, assimilated_ode)

        self.f_true = true_ode


class AsyncSystem(BaseSystem):
    """Class for when the true differential equation is not known and and so may
    not be simulated concurrently with the data assimilated system.

    See `BaseSystem` for additional documentation.
    """
