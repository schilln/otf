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
    _k = 2

    def __init__(self, system: BaseSystem, pre_multistep_solver: BaseSolver):
        """Two-step Adams–Bashforth solver.

        See documentation of `base_solver.MultistepSolver`.

        See https://en.wikipedia.org/wiki/Linear_multistep_method#Two-step_Adams%E2%80%93Bashforth
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
    _k = 4

    def __init__(self, system: BaseSystem, pre_multistep_solver: BaseSolver):
        """Four-step Adams–Bashforth solver.

        See documentation of `base_solver.MultistepSolver`.

        https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods
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

            (true, assimilated), (dt, cs) = vals
            t = true[i - 1]
            a = assimilated[i - 1]

            k1t, k1a = f_t(t), f_a(cs, t, a)
            k2t, k2a = (
                f_t(tmp := t + dt * k1t / 2),
                f_a(
                    cs,
                    tmp,
                    a + dt * k1a / 2,
                ),
            )
            k3t, k3a = (
                f_t(tmp := t + dt * k2t / 2),
                f_a(
                    cs,
                    tmp,
                    a + dt * k2a / 2,
                ),
            )
            k4t, k4a = (
                f_t(tmp := t + dt * k3t),
                f_a(
                    cs,
                    tmp,
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
    def __init__(self, system: BaseSystem, options: dict = dict()):
        """Wrapper around `scipy.integrate.solve_ivp` implementing the same
        external interface as `MultistageSolver`.

        Note that this class does not use or implement all methods defined in
        its parent class since it uses `solve_ivp` (instead of a custom
        implementation of an ODE-solving algorithm using jax).

        See documentation of `MultistageSolver`.

        Parameters
        ----------
        system
            An instance of `BaseSystem` to simulate forward in time.
        options
            Optional arguments that will be passed directly to `solve_ivp`

        Methods
        -------
        solve
            Simulate `self.system` forward in time.

        Attributes
        ----------
        system
            The `system` passed to `__init__`; read-only
        options
            The `options` passed to `__init__`, but may be modified at any time
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
