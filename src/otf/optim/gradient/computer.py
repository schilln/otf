import jax
from jax import numpy as jnp

from ...system.base import BaseSystem
from ..parameter_update_option import UpdateOption
from . import sensitivity

jndarray = jnp.ndarray


class GradientComputer:
    def __init__(
        self,
        system: BaseSystem,
        update_option: UpdateOption = UpdateOption.last_state,
    ):
        self._system = system
        self._update_option = update_option
        self._weight = None

    def compute_gradient(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        raise NotImplementedError()

    def set_weight(self, weight: jndarray | None):
        """Weight the error using a (positive definite) matrix.

        Use this matrix to weight the error (defined as the two-norm of the
        difference between the data assimilated system state and the observed
        portion of true system state). This accordingly weights the derivative
        of the error with respect to the unknown parameters of the data
        assimilated system.

        If assimilating a system with noisy measurements, it is common to use
        the *inverse* of the covariance matrix as the weight.

        Parameters
        ----------
        weight
            Square array. Each dimension should be equal to the number of
            observed variables. Pass `None` to unset the weight (equivalently,
            set the weight to the identity).
        """
        self._weight = weight

    system = property(lambda self: self._system)
    update_option = property(lambda self: self._update_option)


class SensitivityGradient(GradientComputer):
    def __init__(
        self,
        system: BaseSystem,
        update_option: UpdateOption = UpdateOption.last_state,
    ):
        super().__init__(system, update_option)

        match update_option:
            case UpdateOption.last_state:
                self.compute_gradient = self._last_state
            case UpdateOption.mean_state:
                self.compute_gradient = self._mean_state
            case UpdateOption.mean_gradient:
                self.compute_gradient = self._mean_derivative

    def _compute_gradient(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        diff = assimilated[self.system.observed_mask] - observed_true
        w = sensitivity._compute_sensitivity(self.system, assimilated)
        m = w.shape[1]
        if self._weight is None:
            gradient = diff @ w.reshape(-1, m).conj()
        else:
            gradient = diff @ self._weight @ w.reshape(-1, m).conj()
        return gradient if self.system.cs.dtype == complex else gradient.real

    def _last_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        return self._compute_gradient(observed_true[-1], assimilated[-1])

    def _mean_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        return self._compute_gradient(
            observed_true.mean(axis=0), assimilated.mean(axis=0)
        )

    def _mean_derivative(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        return jax.vmap(self._compute_gradient, 0)(
            observed_true, assimilated
        ).mean(axis=0)
