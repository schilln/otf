"""

Classes Implementing Optimization Algorithms
--------------------------------------------
GradientDescent
WeightedLevenbergMarquardt
LevenbergMarquardt
OptaxWrapper
    Wraps a given optax optimizer as an `Optimizer`
"""

import optax
from jax import numpy as jnp

from ..system.base import BaseSystem
from . import gradient
from .base import BaseOptimizer

jndarray = jnp.ndarray


class DummyOptimizer(BaseOptimizer):
    def __init__(
        self,
        system: BaseSystem,
        gradient_computer: gradient.GradientComputer | None = None,
    ):
        """Don't perform parameter updates."""
        super().__init__(system, gradient_computer)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        return jnp.zeros_like(self.system.cs)

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        return jnp.zeros_like(self.system.cs)


class GradientDescent(BaseOptimizer):
    def __init__(
        self,
        system: BaseSystem,
        learning_rate: float = 1e-4,
        gradient_computer: gradient.GradientComputer | None = None,
    ):
        """Perform gradient descent.

        See documentation of `Optimizer`.

        Parameters
        ----------
        learning_rate
            The learning rate to use in gradient descent
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
    def __init__(
        self,
        system: BaseSystem,
        learning_rate: float = 1e-3,
        lam: float = 1e-2,
        gradient_computer: gradient.SensitivityGradient | None = None,
    ):
        """Perform the Levenberg–Marquardt modification of Gauss–Newton.

        Parameters
        ----------
        learning_rate
            The learning rate (scalar by which to multiply the step)
        lam
            Levenberg–Marquardt parameter
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
            Instance of `optax.GradientTransformationExtraArgs`
            For example, `optax.adam(learning_rate=1e-1)`.
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
