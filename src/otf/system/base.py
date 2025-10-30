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

"""
By using slice objects, I already require that the slice be
contiguous. I think the easiest thing would be to require that the
state arrays also be one-dimensional (which can be accomplished for
something like Lorenz '96 via reshape and/or ravel). This would
simplify/make possible using the unobserved portion of the state, and
add only a slight amount of overhead to defining systems.
Or... perhaps the best idea would be to allow slices (and masks) as I
am now (masks should just get jit compiled and be equally fast) but
then if the states aren't flat, then wrap the ODE functions to accept
flattened states and also store a flattened version of the slice/mask
to use when computing unobserved stuff.
"""

"""
I want to allow ODE definitions using arbitrary (non-ragged) shapes. I want to
allow observed slices or masks.

So when self.assimilated_ode is run for the first time, it needs to redefine
itself to
    1. expect a flattened state
    2. reshape the state and pass it into the actual assimilated_ode function
    3. return the flattened the output derivative
Same for self.true_ode. We'll need to compute a separate mask object for the
flattened version of the state. Then we can invert this mask to compute the Qw^0
term in compute_w. Perhaps another mask object for true since true and
assimilated may not even share the same model or state dimension. The only
requirement is that their observed portions be comparable in terms of dimension.

Then self.f_assimilated and self.f_true need to accept shaped inputs (they are
public methods) and then use assimilated_ode and true_observed appropriately,
i.e., reshaping the outputs of assimilated_ode to match.
"""

"""
Game plan

__init__
    1. Set self._shaped_assimilated_ode = assimilated_ode.
    2. Define self._assimilated_ode as a function that on the first run
        a. Stores the shape of the passed-in state in self._assimilated_shape.
        b. Compute the observed and unobserved mask of the flattened version
            of the state using the provided slice/mask.
        c. Redefines itself to
            1. Take a flattened state (this will make autodiff nice)
            2. Reshape the flat input to self._assimilated_shape.
            3. Pass this through self._shaped_assimilated_ode
            4. Flatten the output and return.
        d. Actually runs itself.
    3. Redefine self.f_assimilated to still accept shaped inputs but flatten
        them to pass into self._assimilated_ode.
"""


class BaseSystem:
    """A base class for defining systems of differential equations and to which
    the on-the-fly (OTF) method of data assimilation may be applied.
    """

    def __init__(
        self,
        mu: float,
        gs: jndarray,
        cs: jndarray,
        observed_slice: slice,
        assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
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
        observed_slice
            The slice denoting the observed part of the true and data
            assimilated system states when nudging in `f_assimilated`. May use
            `jnp.s_` to define slice to use. To observed the entire system, use
            `jnp.s_[:]`.
        assimilated_ode
            Function that computes the time derivative of the data assimilated
            state using the current estimated parameters `cs`.
            Parameters: (cs, true_observed, assimilated)
        complex_differentiation
            Set to True if state values may take complex values (e.g., if
            time integrating Fourier coefficients). This allows
            auto-differentiation to work.

        Methods
        -------
        f_assimilated
            Computes the time derivative of the data assimilated state.
        compute_w
            Computes the leading-order approximation of the sensitivity
            equations.
            May be overridden (see docstring).
        """
        self._mu = mu
        self._gs = gs
        self._observed_slice = (
            observed_slice
            if isinstance(observed_slice, tuple)
            else (observed_slice,)
        )
        self._cs = cs
        self._assimilated_ode = assimilated_ode

        self._complex_differentiation = complex_differentiation

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
        s = self.observed_slice

        assimilated_p = self._assimilated_ode(cs, assimilated)
        assimilated_p = assimilated_p.at[s].subtract(
            self.mu * (assimilated[s] - true_observed)
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
            The ith row corresponds to the asymptotic approximation of the ith
            sensitivity corresponding to the ith unknown parameter ci
        """
        return self._compute_w(self.cs, assimilated)

    @partial(jax.jit, static_argnames="self")
    def _compute_w(self, cs: jndarray, assimilated: jndarray) -> jndarray:
        dFdc = jax.jacrev(
            self._assimilated_ode,
            0,
            holomorphic=self.complex_differentiation,
        )(cs, assimilated)
        jax.debug.print("{val}", val=dFdc)
        # jax.debug.breakpoint()
        # jax.debug.print("{val}", val=dFdc[self.observed_slice])
        # jax.debug.print("{val}", val=dFdc[self.observed_slice].T)
        return dFdc[self.observed_slice].T

    def _set_cs(self, cs):
        self._cs = cs

    # The following attributes are read-only.
    mu = property(lambda self: self._mu)
    gs = property(lambda self: self._gs)
    cs = property(lambda self: self._cs, _set_cs)
    observed_slice = property(lambda self: self._observed_slice)
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
        observed_slice: slice,
        assimilated_ode: Callable[[jndarray, jndarray], jndarray],
        true_ode: Callable[[jndarray, jndarray], jndarray],
        complex_differentiation: bool = False,
    ):
        """

        Parameters
        ----------
        See `BaseSystem` for other parameter definitions.

        true_ode
            Function that computes the time derivative of the true state
            Parameters: (gs, true)
        """
        super().__init__(
            mu,
            gs,
            cs,
            observed_slice,
            assimilated_ode,
            complex_differentiation,
        )

        self._true_ode = true_ode

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


class System_ModelUnknown(BaseSystem):
    """Class for when the true differential equation is not known and and so may
    not be simulated concurrently with the data assimilated system.

    See `BaseSystem` for additional documentation.
    """
