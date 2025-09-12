"""Abstract base class for a system of differential equations to use the nudging
approach to data assimilation to nudge an estimated system toward an observed
"ground truth" system.
"""

from functools import partial

import jax
from jax import numpy as jnp

jndarray = jnp.ndarray


class System:
    def __init__(
        self,
        μ: float,
        gs: jndarray,
        bs: jndarray,
        cs: jndarray,
        observed_slice: slice,
    ):
        """
        An abstract base class defining some common methods and an interface
        on which other classes and functions can rely when using with derived
        classes.

        Parameters
        ----------
        μ
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

        Methods
        -------
        f
            Computes the time derivative the true and nudged system, given the
            current states of each and the current estimated parameters for the
            nudged system.
        compute_w
            Computes the leading-order approximation of the sensitivity
            equations.
            May be overridden (see docstring).

        Abstract Methods
        ----------------
        These must be overridden by subclasses.

        ode
            Computes the time derivative of the true system, given its current
            state.
        estimated_ode
            Computes the time derivative of the nudged system, given its current
            state and the current estimate of its parameters.
        """
        self._μ = μ
        self._gs = gs
        self._bs = bs
        self._observed_slice = observed_slice

        self._cs = cs

    def f(
        self,
        cs: jndarray,
        true: jndarray,
        nudged: jndarray,
    ) -> tuple[jndarray, jndarray]:
        """Computes the time derivative of `true` and `nudged` using `ode` and
        `estimated_ode` followed by nudging the nudged system using the observed
        portion of `true`.

        This function will be jitted.

        Parameters
        ----------
        cs
            Estimated parameter values to be used by the nudged system
        true
            True system
        nudged
            Nudged system

        Returns
        -------
        truep, nudgedp
            The time derivatives of `true` and `nudged`
        """
        s = self.observed_slice

        nudgedp = self.estimated_ode(cs, nudged)
        nudgedp = nudgedp.at[s].subtract(self.μ * (nudged[s] - true[s]))

        return self.ode(true), nudgedp

    def ode(
        self,
        true: jndarray,
    ) -> jndarray:
        """Computes the time derivative of `true`.
        This method should be overridden according to the desired differential
        equation.

        This function will be jitted.

        Parameters
        ----------
        true
            True system

        Returns
        -------
        truep
            The time derivative of `true`
        """
        raise NotImplementedError()

    def estimated_ode(
        self,
        cs: jndarray,
        nudged: jndarray,
    ) -> jndarray:
        """Computes the time derivative of `nudged`, using the current estimated
        parameters `cs`.
        This method should be overridden according to the desired differential
        equation.

        This function will be jitted.

        Parameters
        ----------
        cs
            Estimated parameter values to be used by the nudged system
        nudged
            Nudged system

        Returns
        -------
        nudgedp
            The time derivative of `nudged`
        """
        raise NotImplementedError()

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
            senstitivity corresponding to the ith unknown parameter ci
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
