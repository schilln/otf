"""Optimization algorithms for estimating unknown system parameters.

This module provides concrete implementations of `BaseOptimizer` used to update
the `cs` parameters of a `BaseSystem` instance, including simple gradient
descent, Levenberg–Marquardt variants, and an adapter for Optax optimizers.
"""

import optax
from jax import numpy as jnp

from ..system.base import BaseSystem
from . import gradient
from .base import BaseOptimizer

jndarray = jnp.ndarray


class DummyOptimizer(BaseOptimizer):
    """Optimizer that performs no parameter updates (useful for testing)."""

    def __init__(
        self,
        system: BaseSystem,
        gradient_computer: gradient.GradientComputer | None = None,
    ):
        """Initialize the dummy optimizer.

        Parameters
        ----------
        system
            Target `BaseSystem` instance.
        gradient_computer
            Optional `GradientComputer`.
        """
        super().__init__(system, gradient_computer)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        return jnp.zeros_like(self.system.cs)

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        return jnp.zeros_like(self.system.cs)


class GradientDescent(BaseOptimizer):
    """Simple gradient-descent optimizer."""

    def __init__(
        self,
        system: BaseSystem,
        learning_rate: float = 1e-4,
        gradient_computer: gradient.GradientComputer | None = None,
    ):
        """Create a gradient-descent optimizer.

        Parameters
        ----------
        learning_rate
            Scalar learning rate used to scale the negative gradient.
        """
        super().__init__(system, gradient_computer)
        self.learning_rate = learning_rate

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)
        return self.step_from_gradient(gradient, observed_true, nudged)

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        return -self.learning_rate * gradient


class WeightedLevenbergMarquardt(BaseOptimizer):
    """Weighted Levenberg–Marquardt optimizer (Gauss–Newton variant)."""

    def __init__(
        self,
        system: BaseSystem,
        learning_rate: float = 1e-3,
        lam: float = 1e-2,
        gradient_computer: gradient.GradientComputer | None = None,
    ):
        """Perform a weighted version of the Levenberg–Marquardt modification of
        Gauss–Newton.

        Parameters
        ----------
        learning_rate
            The learning rate (scalar by which to multiply the step)
        lam
            Levenberg–Marquardt parameter
        """
        super().__init__(system, gradient_computer)
        self.learning_rate = learning_rate
        self.lam = lam

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)
        return self.step_from_gradient(gradient, observed_true, nudged)

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        mat = jnp.outer(gradient, gradient)

        step = jnp.linalg.solve(
            mat + self.lam * jnp.eye(mat.shape[0]), gradient
        )

        return -self.learning_rate * step


class LevenbergMarquardt(BaseOptimizer):
    """Levenberg–Marquardt optimizer using sensitivity-based gradients."""

    def __init__(
        self,
        system: BaseSystem,
        learning_rate: float = 1e-3,
        lam: float = 1e-2,
        gradient_computer: gradient.SensitivityGradient | None = None,
    ):
        """Levenberg–Marquardt optimizer using sensitivity-based gradients.

        This implementation requires a `SensitivityGradient` instance and is
        currently implemented only for the `UpdateOption.last_state` update
        method of the gradient computer.

        Parameters
        ----------
        learning_rate
            Scalar multiplier applied to the computed step.
        lam
            Levenberg–Marquardt damping parameter.
        """
        if not isinstance(gradient_computer, gradient.SensitivityGradient):
            raise NotImplementedError(
                "not yet implemented for adjoint-based gradient computation"
            )
        if gradient_computer.update_option is not (
            gradient.sensitivity.UpdateOption.last_state
        ):
            raise NotImplementedError(
                "currently implemented only for last state gradient computation"
            )

        super().__init__(system, gradient_computer)
        self.learning_rate = learning_rate
        self.lam = lam

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)
        return self.step_from_gradient(gradient, observed_true, nudged)

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        w = self.gradient_computer._compute_sensitivity_asymptotic(
            self.system, nudged[-1:], self.system.cs
        ).squeeze(axis=0)
        m = w.shape[1]
        w_2d = w.reshape(-1, m)
        mat = jnp.real(w_2d.conj().T @ w_2d)

        step = jnp.linalg.solve(
            mat + self.lam * jnp.eye(mat.shape[0]), gradient
        )

        return -self.learning_rate * step


class OptaxWrapper(BaseOptimizer):
    """Adapter that wraps an Optax optimizer as a `BaseOptimizer`."""

    def __init__(
        self,
        system: BaseSystem,
        optimizer: optax.GradientTransformationExtraArgs,
        gradient_computer: gradient.GradientComputer | None = None,
    ):
        """Wrap a given Optax optimizer.

        Parameters
        ----------
        optimizer
            Instance of `optax.GradientTransformationExtraArgs` For example,
            `optax.adam(learning_rate=1e-1)`.
        """
        super().__init__(system, gradient_computer)
        self.optimizer = optimizer
        self.opt_state = self.optimizer.init(system.cs)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)
        return self.step_from_gradient(gradient, observed_true, nudged)

    def step_from_gradient(
        self, gradient: jndarray, true_observed: jndarray, nudged: jndarray
    ) -> jndarray:
        update, self.opt_state = self.optimizer.update(gradient, self.opt_state)
        return update
