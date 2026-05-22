"""Optimization utilities for parameter estimation during data assimilation.

This module provides base classes and helpers used to update the `cs` parameters
of `BaseSystem` instances during data assimilation. See the individual class and
function docstrings for details and usage examples.
"""

from collections.abc import Callable

import jax
import numpy as np
from jax import numpy as jnp

from ..system.base import BaseSystem
from . import gradient

jndarray = jnp.ndarray


class BaseOptimizer:
    """Base interface for optimizers that estimate `BaseSystem` parameters.

    Subclasses must implement `step(observed_true, nudged)` and may override
    `step_from_gradient`.
    """

    def __init__(
        self,
        system: BaseSystem,
        gradient_computer: gradient.GradientComputer | None = None,
    ):
        """Create a `BaseOptimizer`.

        Parameters
        ----------
        system
            `BaseSystem` instance whose `cs` are to be optimized.
        gradient_computer
            Optional `GradientComputer`; if omitted a sensible default is
            constructed.
        """
        if gradient_computer is None:
            gradient_computer = gradient.SensitivityGradient(system)

        self._system = system
        self._weight = None
        self._gradient_computer = gradient_computer
        self.compute_gradient = self._gradient_computer.compute_gradient

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        """Abstract: compute the parameter update vector.

        Subclasses should return the vector that will be added to `system.cs` to
        obtain the updated parameters.

        Parameters
        ----------
        observed_true
            Observed portion of the true system's state (array-like).
        nudged
            The nudged/assimilated system's state (array-like).

        Returns
        -------
        step
            Vector to add to `system.cs` to obtain the new parameter values.
        """

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        """Optional: compute an update from a precomputed `gradient`.

        Default implementations may delegate to `step`. Subclasses that have
        closed-form updates from the gradient can override this for speed and
        clarity.

        Parameters
        ----------
        gradient
            Derivative of the assimilation error with respect to parameters
            (same shape as `system.cs`).
        observed_true
            Observed portion of the true system's state (array-like).
        nudged
            The nudged/assimilated system's state (array-like).

        Returns
        -------
        step
            Vector to add to `system.cs` to obtain the new parameter values.
        """

    def __call__(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        """Compute the new parameter values following one step of the
        optimization algorithm.

        Parameters
        ----------
        observed_true
            The observed portion of the true system's state
        nudged
            The nudged system's state

        Returns
        -------
        new_cs
            The new values for `system.cs`
        """
        return self.system.cs + self.step(observed_true, nudged)

    # The following attributes are read-only.
    system = property(lambda self: self._system)
    gradient_computer = property(lambda self: self._gradient_computer)


class PartialOptimizer(BaseOptimizer):
    """Wrap an optimizer to update only a subset of parameters.

    The wrapped optimizer computes a full update vector; `PartialOptimizer`
    masks that update so only selected indices in `system.cs` are changed.
    """

    def __init__(
        self,
        optimizer: BaseOptimizer,
        param_idx: jndarray | None = None,
    ):
        """Initialize the wrapper.

        Parameters
        ----------
        optimizer
            Base optimizer used to compute the full parameter update.
        param_idx
            Indices of `system.cs` that should be updated. Provide an explicit
            array-like index (e.g. `np.array([0, 2])`).
        """
        # Define the attributes that belong to this class (versus those of the
        # wrapped class) so they can be distinguished and routed properly.
        super().__setattr__(
            "_own_attrs", {"_system", "system", "optimizer", "mask"}
        )
        super().__init__(optimizer.system)

        self.optimizer = optimizer

        n = len(self.optimizer.system.cs)
        self.mask = jnp.zeros(n, dtype=bool)
        self.mask = self.mask.at[param_idx].set(True)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        update = self.optimizer.step(observed_true, nudged)
        return jnp.where(self.mask, update, 0)

    def __getattr__(self, name):
        """For attributes that aren't defined in this class, route access to the
        wrapped optimizer.
        """
        return getattr(self.optimizer, name)

    def __setattr__(self, name, value):
        """For attributes that aren't defined in this class, route access to the
        wrapped optimizer.
        """
        if name in self._own_attrs:
            super().__setattr__(name, value)
        else:
            setattr(self.optimizer, name, value)


class Regularizer(BaseOptimizer):
    """Produce parameter-space penalties used as a regularization term.

    The `step` method returns the negative derivative of the regularization
    penalty (same shape as `system.cs`) and can be combined with a
    gradient-based optimizer.
    """

    def __init__(
        self,
        system: BaseSystem,
        ord: int | float | Callable,
        prior: jndarray | None = None,
        callable_is_derivative: bool | None = None,
    ):
        """Initialize a `Regularizer`.

        Parameters
        ----------
        ord
            If an int/float, interpreted as the `ord`-norm penalty (1 or 2
            supported explicitly). If a callable, see `callable_is_derivative`.
        prior
            Optional prior parameter vector; penalty is applied to `(cs -
            prior)`. If omitted, the prior is taken to be zero.
        callable_is_derivative
            If `ord` is callable, set to True when `ord` already returns the
            derivative (an array shaped like `system.cs`); set to False when
            `ord` returns a scalar penalty and its derivative should be
            auto-differentiated.
        """
        if prior is None:
            self._prior = jnp.zeros_like(system.cs)
        else:
            if prior.shape != system.cs.shape:
                raise ValueError(
                    "`prior` should have same shape as `system.cs`"
                )
            self._prior = prior

        match ord:
            case int() | float():
                pass
            case Callable() if callable_is_derivative is None:
                raise ValueError(
                    "`callable_is_derivative` must be a bool when `ord` is a "
                    "callable"
                )
            case Callable() if callable_is_derivative:
                if ord(system.cs).shape != system.cs.shape:
                    raise ValueError(
                        "`ord` must return an array of the same shape as the "
                        "parameters `system.cs`"
                    )
            case Callable() if not callable_is_derivative:
                if not jnp.isscalar(ord(system.cs)):
                    raise ValueError(
                        "`ord` must be scalar-valued since "
                        "`callable_is_derivative` is False"
                    )
            case _:
                raise ValueError("`ord` is an invalid type")

        super().__init__(system)
        self._ord = ord
        self._callable_is_derivative = callable_is_derivative

    def step(self, *_):
        """Return the negative derivative of the regularization penalty.

        The returned array has the same shape as `system.cs` and can be added to
        a gradient-based update.
        """
        ord, prior = self.ord, self.prior
        cs = self.system.cs
        match ord:
            case 2:
                return -2 * (cs - prior)
            case 1:
                return -jnp.sign(cs - prior)
            case int() | float():
                # FutureFIXME: Evaluating at `cs - prior` might not be right.
                return -jax.jacfwd(lambda ps: jnp.norm(ps, ord=ord))(cs - prior)
            case Callable() if self.callable_is_derivative:
                # FutureFIXME: Evaluating at `cs - prior` might not be right.
                return -ord(cs - prior)
            case Callable() if not self.callable_is_derivative:
                # FutureFIXME: Evaluating at `cs - prior` might not be right.
                return -jax.jacfwd(ord, holomorphic=True)(cs - prior)
            case _:
                raise ValueError("`self.ord` is no longer a valid value")

    ord = property(lambda self: self._ord)
    callable_is_derivative = property(lambda self: self._callable_is_derivative)
    prior = property(lambda self: self._prior)


class OptimizerChain(BaseOptimizer):
    """Combine several optimizers by summing their weighted updates."""

    def __init__(
        self,
        system: BaseSystem,
        learning_rate: float,
        optimizers: list[BaseOptimizer],
        weights: list[float],
    ):
        """Initialize an `OptimizerChain`.

        Parameters
        ----------
        learning_rate
            Scalar applied to the total weighted update (controls overall step
            size).
        optimizers
            Sequence of `BaseOptimizer` instances whose `step` results will be
            combined. All optimizers should target the same `system`.
        weights
            Sequence of relative weights (one per optimizer). These are
            normalized internally so their sum is one.

        Notes
        -----
        It can be convenient to set each individual optimizer's internal
        learning rate to 1.0 and control the combined step size with
        `learning_rate` supplied here.
        """
        assert len(optimizers) == len(weights), (
            "`optimizers` and `weights` should have same length"
        )

        super().__init__(system)
        self.learning_rate = learning_rate
        self._optimizers = optimizers
        self._weights = jnp.array(weights) / sum(weights)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        return self.learning_rate * sum(
            [
                weight * optimizer.step(observed_true, nudged)
                for weight, optimizer in zip(self.weights, self.optimizers)
            ]
        )

    optimizers = property(lambda self: self._optimizers)
    weights = property(lambda self: self._weights)


def pruned_factory(system_type: type[BaseSystem]) -> type[BaseSystem]:
    """Return a 'pruned' variant of `system_type`.

    If a parameter in `cs` of the system is to be set below its corresponding
    threshold (in absolute value), it will be set to zero permanently.
    Optionally require that this occur at least a specified number of times
    consecutively before setting a parameter to zero permanently.

    Parameters
    ----------
    system_type
        The type of `BaseSystem` (not an instance) to be wrapped, e.g.,
        `system.System_ModelKnown`
    """

    class Pruned(system_type):
        def __init__(
            self,
            *args,
            threshold: float | jndarray | np.ndarray,
            iterations: int | jndarray | np.ndarray | None = None,
            **kwargs,
        ):
            """

            Parameters
            ----------
            threshold
                If a float, all parameters `cs` are compared against this common
                value. If an array, each parameter is compared against the value
                in `cs` in the same position. To disable pruning for a
                parameter, set its threshold to zero.
            iterations
                Require each parameter to be less than its corresponding
                threshold at least `iterations` times consecutively before
                setting it to zero (permanently). As with `threshold`, if an
                `int`, each parameter will use this common value, but if an
                array, then each parameter will use its corresponding value. If
                None, only one time being less than `threshold` is needed to set
                a parameter to zero.
            """
            super().__init__(*args, **kwargs)

            if isinstance(threshold, (jndarray, np.ndarray)):
                if self._cs.shape != threshold.shape:
                    raise ValueError(
                        "`threshold` must have same shape as `system.cs`"
                    )
            self.threshold = np.array(threshold)

            if isinstance(iterations, (jndarray, np.ndarray)):
                if self._cs.shape != iterations.shape:
                    raise ValueError(
                        "`iterations` must have same shape as `system.cs`"
                    )
            self.iterations = (
                None if iterations is None else np.array(iterations)
            )

            # A mask in which True indicates the corresponding parameter should
            # be set to zero.
            self._set_zero = np.zeros_like(self.cs, dtype=bool)

            # Count the number of times each parameter is below its threshold in
            # a row.
            self._counter = np.zeros_like(self.cs, dtype=int)

        def _set_cs(self, cs):
            # For parameters under the threshold, set the mask to True. Don't
            # change the mask where it already was True.
            below_threshold = np.abs(self.cs) < self.threshold

            # Increment the counter to parameters below their threshold and
            # reset to zero the counter for parameters not below their
            # threshold.
            if self.iterations is not None:
                self._counter += below_threshold
                self._counter[~below_threshold] = 0
                at_least_counter = self._counter >= self.iterations
            else:
                at_least_counter = True

            set_zero = below_threshold & at_least_counter
            self._set_zero[set_zero] = True
            self._cs = jnp.where(self._set_zero, 0, cs)

            # Reset the counter for parameters already set to zero (no point
            # continuing to count).
            if self.iterations is not None:
                self._counter[self._set_zero] = 0

        cs = property(
            lambda self: self._cs, lambda self, value: self._set_cs(value)
        )

    # TODO: Check if this is working correctly. Or rework how pruned systems are
    # constructed.
    doc = (
        "This is a 'Pruned' version of the original class "
        f"({system_type.__module__}.{system_type.__qualname__}); "
        "that is, if a parameter in `self.cs` is to be set below its "
        "corresponding threshold (in absolute value), it will be set to zero "
        "permanently."
    )

    Pruned.__module__ = system_type.__module__
    Pruned.__name__ = f"{system_type.__name__}_Pruned"
    Pruned.__qualname__ = system_type.__qualname__
    Pruned.__doc__ = (
        doc
        if system_type.__doc__ is None
        else system_type.__doc__ + "\n\n" + doc
    )
    Pruned.__annotations__ = system_type.__annotations__

    return Pruned
