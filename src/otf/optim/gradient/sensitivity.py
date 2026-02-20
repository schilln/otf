from collections.abc import Callable
from enum import Enum
from functools import partial

import jax
from jax import numpy as jnp

from ...system.base import BaseSystem, System_ModelUnknown
from ...time_integration.base import MultistepSolver, SinglestepSolver
from .gradient_computer import GradientComputer

jndarray = jnp.ndarray


class UpdateOption(Enum):
    """Enum for selecting parameter update methods.

    Options include:

    - last_state: Uses the last observed state for the update and the asymptotic
      approximation for the observed portion of the sensitivity.
    - mean_state: Uses the mean of the observed states for the update and the
      asymptotic approximation for the observed portion of the sensitivity.
    - mean_gradient: Uses the mean of the gradients for the update and the
      asymptotic approximation for the observed portion of the sensitivity.
    - complete: Uses the mean of the gradients for the update and the complete
      sensitivity via simulation.
    """

    last_state = 0
    mean_state = 1
    mean_gradient = 2
    complete = 3


class SensitivityGradient(GradientComputer):
    def __init__(
        self,
        system: BaseSystem,
        update_option: UpdateOption = UpdateOption.last_state,
        solver: tuple[type[SinglestepSolver | MultistepSolver]]
        | type[SinglestepSolver | MultistepSolver]
        | None = None,
        dt: float | None = None,
        use_unobserved_asymptotics: bool = False,
    ):
        """
        use_unobserved_asymptotics
            Set to true to attempt to use additional asymptotic information from
            "unobserved" portion of simulated state. This doesn't have full
            mathematical support.
        """
        super().__init__(system)

        self._update_option = update_option
        self.compute_gradient = self._set_up_gradient(update_option)

        if update_option is UpdateOption.complete:
            if not system.observe_all:
                raise NotImplementedError
            if dt is None:
                raise ValueError("`dt` must not be None for this update option")
            self._dt = dt
            sensitivity_system = SensitivitySystem(system)
            self._solver = self._set_up_solver(sensitivity_system, solver)

        self._use_unobserved_asymptotics = use_unobserved_asymptotics

    def _set_up_gradient(self, update_option: UpdateOption) -> Callable:
        match update_option:
            case UpdateOption.last_state:
                return self._last_state
            case UpdateOption.mean_state:
                return self._mean_state
            case UpdateOption.mean_gradient:
                return self._mean_derivative
            case UpdateOption.complete:
                return self._complete
            case _:
                raise NotImplementedError("update option is not supported")

    def _set_up_solver(
        self,
        system: BaseSystem,
        solver: tuple[type[SinglestepSolver | MultistepSolver]]
        | type[SinglestepSolver | MultistepSolver],
    ) -> SinglestepSolver | MultistepSolver:
        """Initialize and chain solvers for the given system."""
        if not isinstance(solver, tuple):
            solver = (solver,)

        _solver = solver[0](system)
        for s in solver[1:]:
            _solver = s(system, _solver)

        return _solver

    def _last_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        sensitivity = self._compute_sensitivity_asymptotic(
            self, assimilated[-1:], self.system.cs
        )

        return self._compute_gradient(
            observed_true[-1:], assimilated[-1:], self.system.cs, sensitivity
        )

    def _mean_state(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        assimilated_mean = assimilated.mean(axis=0, keepdims=True)
        sensitivity = self._compute_sensitivity_asymptotic(
            self, assimilated_mean, self.system.cs
        )

        return self._compute_gradient(
            observed_true.mean(axis=0),
            assimilated_mean,
            self.system.cs,
            sensitivity,
        )

    def _mean_derivative(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        sensitivity = self._compute_sensitivity_asymptotic(
            self, assimilated, self.system.cs
        )

        return self._compute_gradient(
            observed_true, assimilated, self.system.cs, sensitivity
        )

    def _complete(
        self, observed_true: jndarray, assimilated: jndarray
    ) -> jndarray:
        sensitivity = self._compute_sensitivity_complete(self, assimilated)

        return self._compute_gradient(
            observed_true, assimilated, self.system.cs, sensitivity
        )

    @partial(jax.jit, static_argnames=("self",))
    def _compute_gradient(
        self,
        observed_true: jndarray,
        assimilated: jndarray,
        cs: jndarray,
        sensitivity: jndarray,
    ) -> jndarray:
        diff = assimilated[:, self.system.observed_mask] - observed_true
        if self._weight is None:
            gradient = (
                (jnp.expand_dims(diff, 1) @ sensitivity.conj())
                .squeeze(axis=1)
                .mean(axis=0)
            )
        else:
            gradient = (
                (jnp.expand_dims(diff, 1) @ self._weight @ sensitivity.conj())
                .squeeze(axis=1)
                .mean(axis=0)
            )
        return gradient if cs.dtype == complex else gradient.real

    @staticmethod
    @partial(jax.jit, static_argnames="self")
    def _compute_sensitivity_asymptotic(
        self, assimilated: jndarray, cs: jndarray
    ) -> jndarray:
        """Compute the leading-order approximation of the sensitivity equations.

        Parameters
        ----------
        assimilated
            Data assimilated system state
        cs
            System parameters

        Returns
        -------
        sensitivity
            The ith column corresponds to the asymptotic approximation of the
            ith sensitivity (i.e., w_i = dv/dc_i corresponding to the ith
            unknown parameter c_i)
        """
        s = self.system
        om = s.observed_mask

        df_dc = jax.vmap(s.df_dc, (None, 0))(cs, assimilated)

        if s.observe_all:
            return df_dc / s.mu
        elif not self._use_unobserved_asymptotics:
            return df_dc[:, om] / s.mu
        else:
            df_dv_QW0 = _solve_unobserved(assimilated, cs, df_dc, s)
            return (df_dc[:, om] + df_dv_QW0[:, om]) / s.mu

    @staticmethod
    def _compute_sensitivity_complete(self, assimilated: jndarray) -> jndarray:
        tn, n = assimilated.shape
        tf = self._dt * tn
        m = self.system.cs.shape[0]

        sensitivity0 = jnp.zeros((n, m), dtype=assimilated.dtype).ravel()

        sensitivity, _ = self._solver.solve_assimilated(
            sensitivity0, self._dt, tf, self._dt, assimilated
        )
        return sensitivity.reshape(-1, n, m)

    update_option = property(lambda self: self._update_option)


class SensitivitySystem(System_ModelUnknown):
    def __init__(self, system: BaseSystem):
        super().__init__(
            None, None, system.observed_mask, system.assimilated_ode
        )
        self._system = system

        self._n = len(self.observed_mask)
        self._m = self._system.cs.shape[0]

    def f_assimilated(
        self,
        cs: jndarray,
        assimilated: jndarray,
        sensitivity: jndarray,
    ) -> jndarray:
        sensitivity = sensitivity.reshape(self._n, self._m)
        val = -self.df_dv_fn(cs, assimilated) @ sensitivity + self.df_dc_fn(
            cs, assimilated
        )
        val = val.at[self.observed_mask].subtract(
            self.mu * sensitivity[self.observed_mask]
        )
        return val.ravel()

    mu = property(lambda self: self._system.mu)
    cs = property(lambda self: self._system.cs)
    observed_mask = property(lambda self: self._system.observed_mask)
    df_dv_fn = property(lambda self: self._system.df_dv)
    df_dc_fn = property(lambda self: self._system.df_dc)


@partial(jax.jit, static_argnames="system")
def _solve_unobserved(
    assimilated: jndarray, cs: jndarray, df_dc: jndarray, system: BaseSystem
) -> jndarray:
    s = system
    um = s.unobserved_mask

    df_dv = jax.vmap(s.df_dv, (None, 0))(cs, assimilated)

    QW0 = jax.vmap(jnp.linalg.lstsq)(df_dv[:, um][:, :, um], -df_dc[:, um])[0]

    return df_dv[:, :, um] @ QW0
