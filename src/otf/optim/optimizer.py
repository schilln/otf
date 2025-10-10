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
from .base import BaseOptimizer

jndarray = jnp.ndarray


class DummyOptimizer(BaseOptimizer):
    def __init__(self, system: BaseSystem):
        """Don't perform parameter updates."""
        super().__init__(system)

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        return jnp.zeros_like(self.system.cs)

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        return jnp.zeros_like(self.system.cs)


class GradientDescent(BaseOptimizer):
    def __init__(self, system: BaseSystem, learning_rate: float = 1e-4):
        """Perform gradient descent.

        See documentation of `Optimizer`.

        Parameters
        ----------
        learning_rate
            The learning rate to use in gradient descent
        """
        super().__init__(system)
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
        self, system: BaseSystem, learning_rate: float = 1e-3, lam: float = 1e-2
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
        super().__init__(system)
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
        self, system: BaseSystem, learning_rate: float = 1e-3, lam: float = 1e-2
    ):
        """Perform the Levenberg–Marquardt modification of Gauss–Newton.

        Parameters
        ----------
        learning_rate
            The learning rate (scalar by which to multiply the step)
        lam
            Levenberg–Marquardt parameter
        """
        super().__init__(system)
        self.learning_rate = learning_rate
        self.lam = lam

    def step(self, observed_true: jndarray, nudged: jndarray) -> jndarray:
        gradient = self.compute_gradient(observed_true, nudged)
        return self.step_from_gradient(gradient, observed_true, nudged)

    def step_from_gradient(
        self, gradient: jndarray, observed_true: jndarray, nudged: jndarray
    ) -> jndarray:
        w = self.system.compute_w(nudged)
        m = w.shape[0]
        w_flat = w.reshape(m, -1)
        mat = jnp.real(w_flat.conj() @ w_flat.T)

        step = jnp.linalg.solve(
            mat + self.lam * jnp.eye(mat.shape[0]), gradient
        )

        return -self.learning_rate * step


class OptaxWrapper(BaseOptimizer):
    def __init__(
        self,
        system: BaseSystem,
        optimizer: optax.GradientTransformationExtraArgs,
    ):
        """Wrap a given Optax optimizer.

        Parameters
        ----------
        optimizer
            Instance of `optax.GradientTransformationExtraArgs`
            For example, `optax.adam(learning_rate=1e-1)`.
        """
        super().__init__(system)
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
