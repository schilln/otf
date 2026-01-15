from jax import numpy as jnp

from ...system.base import BaseSystem
from ..parameter_update_option import UpdateOption

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
