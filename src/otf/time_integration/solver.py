"""Explicit and wrapped ODE solvers for time integration.

This module provides a small collection of time-integration algorithms used
by the `otf` package: simple single-step and multistage integrators (e.g.
`ForwardEuler`, `RK4`), multistep Adams methods, and a `SolveIvp` wrapper
around `scipy.integrate.solve_ivp` that matches the solver interface used in
this package.
"""

from functools import partial

import scipy
from jax import jit
from jax import numpy as jnp

from ..system import BaseSystem
from .base import (
    BaseSolver,
    MultistageSolver,
    MultistepSolver,
    SinglestepSolver,
)

jndarray = jnp.ndarray


class ForwardEuler(SinglestepSolver):
    """Forward Euler solver.

    See documentation of `SinglestepSolver`.
    """

    def _step_factory(self):
        def step_true(i, vals):
            f = self.system.f_true

            true, (dt,) = vals
            t = true[i - 1]

            t = t.at[:].add(dt * f(t))

            true = true.at[i].set(t)

            return true, (dt,)

        def step_assimilated(i, vals):
            f = self.system.f_assimilated

            assimilated, (dt, cs, true_observed) = vals
            t = true_observed[i - 1]
            a = assimilated[i - 1]

            a = a.at[:].add(dt * f(cs, t, a))

            assimilated = assimilated.at[i].set(a)

            return assimilated, (dt, cs, true_observed)

        return step_true, step_assimilated


class TwoStepAdamsBashforth(MultistepSolver):
    """Two-step explicit Adams–Bashforth multistep integrator.

    This class implements the classical two-step Adams–Bashforth method for
    explicit integration of the nonlinear terms. A `pre_multistep_solver` may
    be provided to generate initial steps when starting the integration. If
    `pre_multistep_solver` is `None`, callers must supply the required two
    initial history states when invoking `solve`/`solve_true`.
    """

    _k = 2

    def __init__(
        self, system: BaseSystem, pre_multistep_solver: BaseSolver | None = None
    ):
        """Initialize the two-step Adams–Bashforth solver.

        Parameters
        ----------
        system
            The system to integrate.
        pre_multistep_solver
            Optional solver used to produce initial steps until two history
            values are available. If `None`, callers must provide the two
            initial states when starting integration.
        """

        super().__init__(system, pre_multistep_solver)

    def _step_factory(self):
        def step_true(i, vals):
            f = self.system.f_true

            true, (dt,) = vals
            t2 = true[i - 2]
            t1 = true[i - 1]

            tmp2 = f(t2)
            tmp1 = f(t1)

            t1 = t1.at[:].add(3 / 2 * dt * tmp1 - 1 / 2 * dt * tmp2)

            true = true.at[i].set(t1)

            return true, (dt,)

        def step_assimilated(i, vals):
            f = self.system.f_assimilated

            assimilated, (dt, cs, true_observed) = vals
            t2, a2 = true_observed[i - 2], assimilated[i - 2]
            t1, a1 = true_observed[i - 1], assimilated[i - 1]

            tmp2 = f(cs, t2, a2)
            tmp1 = f(cs, t1, a1)

            a1 = a1.at[:].add(3 / 2 * dt * tmp1 - 1 / 2 * dt * tmp2)

            assimilated = assimilated.at[i].set(a1)

            return assimilated, (dt, cs, true_observed)

        return step_true, step_assimilated


class FourStepAdamsBashforth(MultistepSolver):
    """Four-step Adams–Bashforth explicit multistep integrator.

    Requires a `pre_multistep_solver` to generate the first three steps when
    beginning integration, or callers must supply the initial history of
    four states. `pre_multistep_solver` may be passed as `None` in which case
    the caller is responsible for providing the required initial history when
    invoking `solve`/`solve_true`.
    """

    _k = 4

    def __init__(
        self, system: BaseSystem, pre_multistep_solver: BaseSolver | None = None
    ):
        """Initialize the four-step Adams–Bashforth solver.

        Parameters
        ----------
        system
            The system to integrate.
        pre_multistep_solver
            Optional solver used to produce initial steps until four history
            values are available. If `None`, callers must provide the four
            initial states when starting integration.
        """

        super().__init__(system, pre_multistep_solver)

    def _step_factory(self):
        def step_true(i, vals):
            f = self.system.f_true

            true, (dt,) = vals
            t4 = true[i - 4]
            t3 = true[i - 3]
            t2 = true[i - 2]
            t1 = true[i - 1]

            p4 = f(t4)
            p3 = f(t3)
            p2 = f(t2)
            p1 = f(t1)

            t1 = t1.at[:].add(dt / 24 * (55 * p1 - 59 * p2 + 37 * p3 - 9 * p4))

            true = true.at[i].set(t1)

            return true, (dt,)

        def step_assimilated(i, vals):
            f = self.system.f_assimilated

            assimilated, (dt, cs, true_observed) = vals
            t4, a4 = true_observed[i - 4], assimilated[i - 4]
            t3, a3 = true_observed[i - 3], assimilated[i - 3]
            t2, a2 = true_observed[i - 2], assimilated[i - 2]
            t1, a1 = true_observed[i - 1], assimilated[i - 1]

            p4 = f(cs, t4, a4)
            p3 = f(cs, t3, a3)
            p2 = f(cs, t2, a2)
            p1 = f(cs, t1, a1)

            a1 = a1.at[:].add(dt / 24 * (55 * p1 - 59 * p2 + 37 * p3 - 9 * p4))

            assimilated = assimilated.at[i].set(a1)

            return assimilated, (dt, cs, true_observed)

        return step_true, step_assimilated


class RK4(MultistageSolver):
    """4th-order Runge–Kutta solver.

    See documentation of `base_solver.SinglestepSolver`.

    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    def _step_factory(self):
        def step_true(i, vals):
            f_t = self.system.f_true

            true, (dt,) = vals
            t = true[i - 1]

            k1t = f_t(t)
            k2t = f_t(t + dt * k1t / 2)
            k3t = f_t(t + dt * k2t / 2)
            k4t = f_t(t + dt * k3t)

            t = t.at[:].add((dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t))

            true = true.at[i].set(t)

            return true, (dt,)

        def step(i, vals):
            f_t = self.system.f_true
            f_a = self.system.f_assimilated

            m = self.system.true_observed_mask

            (true, assimilated), (dt, cs) = vals
            t = true[i - 1]
            a = assimilated[i - 1]

            k1t, k1a = f_t(t), f_a(cs, t[m], a)
            k2t, k2a = (
                f_t(tmp := t + dt * k1t / 2),
                f_a(
                    cs,
                    tmp[m],
                    a + dt * k1a / 2,
                ),
            )
            k3t, k3a = (
                f_t(tmp := t + dt * k2t / 2),
                f_a(
                    cs,
                    tmp[m],
                    a + dt * k2a / 2,
                ),
            )
            k4t, k4a = (
                f_t(tmp := t + dt * k3t),
                f_a(
                    cs,
                    tmp[m],
                    a + dt * k3a,
                ),
            )

            t = t.at[:].add((dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t))
            a = a.at[:].add((dt / 6) * (k1a + 2 * k2a + 2 * k3a + k4a))

            true = true.at[i].set(t)
            assimilated = assimilated.at[i].set(a)

            return (true, assimilated), (dt, cs)

        return step_true, step


class SolveIvp(MultistageSolver):
    """`scipy.integrate.solve_ivp` wrapper matching the solver interface.

    This adapter lets users employ SciPy's `solve_ivp` while keeping the same
    external `solve`/`solve_true` signatures used across other solvers in this
    package. Note that integration is performed in NumPy/SciPy (not JAX).
    """

    def __init__(self, system: BaseSystem, options: dict = dict()):
        """Initialize the SolveIvp adapter.

        Parameters
        ----------
        system
            An instance of `BaseSystem` to simulate forward in time.
        options
            Optional keyword arguments that are passed directly to
            `scipy.integrate.solve_ivp`.

        Notes
        -----
        Integration is performed by SciPy (NumPy), not JAX. The `options`
        dictionary may be modified after initialization.
        """

        self._system = system
        self.options = options

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        self._true_shape = true0.shape

        tls = t0 + jnp.arange(round((tf - t0) / dt)) * dt

        result = scipy.integrate.solve_ivp(
            self._ode_true,
            (t0, tf),
            true0,
            t_eval=tls,
            **self.options,
        )

        true = result.y.reshape(*self._true_shape, -1)
        return true.T, tls

    def solve(
        self,
        true0: jndarray,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray]:
        self._true_shape = true0.shape
        self._assimilated_shape = assimilated0.shape

        # The index at which data assimilated states start (to be used in
        # `_unpack` and `_unpack_sequence`)
        self._assimilated_idx = true0.size

        s0 = self._pack(true0, assimilated0)
        tls = t0 + jnp.arange(round((tf - t0) / dt)) * dt

        result = scipy.integrate.solve_ivp(
            self._ode,
            (t0, tf),
            s0,
            t_eval=tls,
            args=(self.system.cs,),
            **self.options,
        )

        true, assimilated = self._unpack_sequence(result.y)
        return true.T, assimilated.T, tls

    @partial(jit, static_argnames="self")
    def _ode_true(self, _, s: jndarray):
        """Wrap `self.system.f_true` using the interface that `solve_ivp`
        expects.
        """
        true = s.reshape(self._true_shape)

        return self.system.f_true(true).ravel()

    @partial(jit, static_argnames="self")
    def _ode(self, _, s: jndarray, cs):
        """Wrap `self.system.f_true` and `self.system.f_assimilated` together
        using the interface that `solve_ivp` expects.
        """
        true, assimilated = self._unpack(s)

        return self._pack(
            self.system.f_true(true),
            self.system.f_assimilated(cs, true, assimilated),
        )

    @partial(jit, static_argnames="self")
    def _pack(self, true: jndarray, assimilated: jndarray):
        """Pack true and data assimilated states into one array for use in
        `solve_ivp`.
        """
        return jnp.concatenate([true.ravel(), assimilated.ravel()])

    @partial(jit, static_argnames="self")
    def _unpack(self, s: jndarray):
        """Unpack true and data assimilated states to use with
        `self.system.f_true` and `self.system.f_assimilated`.
        """
        true = s[: self._assimilated_idx]
        assimilated = s[self._assimilated_idx :]

        return (
            true.reshape(self._true_shape),
            assimilated.reshape(self._assimilated_shape),
        )

    @partial(jit, static_argnames="self")
    def _unpack_sequence(self, s: jndarray):
        """Unpack sequences of true and data assimilated states (e.g., from the
        result of `solve_ivp`).
        """
        true = s[: self._assimilated_idx]
        assimilated = s[self._assimilated_idx :]

        return (
            true.reshape(*self._true_shape, -1),
            assimilated.reshape(*self._assimilated_shape, -1),
        )
