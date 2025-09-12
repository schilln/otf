"""Concrete implementations of `base_solver.Solver`.

Classes
-------
RK4
    Runge–Kutta 4
TwoStepAdamsBashforth
SolveIvp
    Wraps scipy's `solve_ivp`
"""

from functools import partial

from jax import numpy as jnp, jit
import scipy

from base_system import System
from base_solver import Solver, SinglestepSolver, MultistepSolver

jndarray = jnp.ndarray


class RK4(SinglestepSolver):
    """4th-order Runge–Kutta solver.

    See documentation of `base_solver.SinglestepSolver`.

    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    def _step_factory(self):
        def step(i, vals):
            f = self.system.f

            (true, nudged), (dt, cs) = vals
            t = true[i - 1]
            n = nudged[i - 1]

            k1t, k1n = f(cs, t, n)
            k2t, k2n = f(
                cs,
                t + dt * k1t / 2,
                n + dt * k1n / 2,
            )
            k3t, k3n = f(
                cs,
                t + dt * k2t / 2,
                n + dt * k2n / 2,
            )
            k4t, k4n = f(
                cs,
                t + dt * k3t,
                n + dt * k3n,
            )

            t = t.at[:].add((dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t))
            n = n.at[:].add((dt / 6) * (k1n + 2 * k2n + 2 * k3n + k4n))

            true = true.at[i].set(t)
            nudged = nudged.at[i].set(n)

            return (true, nudged), (dt, cs)

        return step


class TwoStepAdamsBashforth(MultistepSolver):
    def __init__(self, system: System, pre_multistep_solver: Solver):
        """Two-step Adams–Bashforth solver.

        See documentation of `base_solver.MultistepSolver`.

        See https://en.wikipedia.org/wiki/Linear_multistep_method#Two-step_Adams%E2%80%93Bashforth
        """
        super().__init__(system, pre_multistep_solver, 2)

    def _step_factory(self):
        def step(i, vals):
            f = self.system.f

            (true, nudged), (dt, cs) = vals
            t2, n2 = true[i - 2], nudged[i - 2]
            t1, n1 = true[i - 1], nudged[i - 1]

            tmp2 = f(cs, t2, n2)
            tmp1 = f(cs, t1, n1)

            t1 = t1.at[:].add(3 / 2 * dt * tmp1[0] - 1 / 2 * dt * tmp2[0])
            n1 = n1.at[:].add(3 / 2 * dt * tmp1[1] - 1 / 2 * dt * tmp2[1])

            true = true.at[i].set(t1)
            nudged = nudged.at[i].set(n1)

            return (true, nudged), (dt, cs)

        return step


class SolveIvp(SinglestepSolver):
    def __init__(self, system: System, options: dict = dict()):
        """Wrapper around `scipy.integrate.solve_ivp` implementing the same
        external interface as `base_solver.SinglestepSolver`.

        Note that this class does not use or implement all methods defined in
        its parent class since it uses `solve_ivp` (instead of a custom
        implementation of an ODE-solving algorithm using jax).

        See documentation of `base_solver.SinglestepSolver`.

        Parameters
        ----------
        system
            An instance of `base_system.System` to simulate forward in time.
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

    def solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        options: dict = dict(),
    ) -> tuple[jndarray, jndarray, jndarray]:
        self._true_shape = true0.shape
        self._nudged_shape = nudged0.shape

        # The index at which nudged states start (to be used in `_unpack` and
        # `_unpack_sequence`)
        self._nudged_idx = true0.size

        s0 = self._pack(true0, nudged0)
        tls = t0 + jnp.arange(round((tf - t0) / dt)) * dt

        result = scipy.integrate.solve_ivp(
            self._ode,
            (t0, tf),
            s0,
            t_eval=tls,
            args=(self.system.cs,),
            **self.options,
        )

        true, nudged = self._unpack_sequence(result.y)
        return true.T, nudged.T, tls

    @partial(jit, static_argnames="self")
    def _ode(self, _, s: jndarray, cs):
        """Wrap `self.system.f` using the interface that `solve_ivp` expects."""
        true, nudged = self._unpack(s)

        return self._pack(*self.system.f(cs, true, nudged))

    @partial(jit, static_argnames="self")
    def _pack(self, true: jndarray, nudged: jndarray):
        """Pack true and nudged states into one array for use in `solve_ivp`."""
        return jnp.concatenate([true.ravel(), nudged.ravel()])

    @partial(jit, static_argnames="self")
    def _unpack(self, s: jndarray):
        """Unpack true and nudged states to use with `self.system.f`."""
        true = s[: self._nudged_idx]
        nudged = s[self._nudged_idx :]

        return (
            true.reshape(self._true_shape),
            nudged.reshape(self._nudged_shape),
        )

    @partial(jit, static_argnames="self")
    def _unpack_sequence(self, s: jndarray):
        """Unpack sequences of true and nudged states (e.g., from the result of
        `solve_ivp`).
        """
        true = s[: self._nudged_idx]
        nudged = s[self._nudged_idx :]

        return (
            true.reshape(*self._true_shape, -1),
            nudged.reshape(*self._nudged_shape, -1),
        )
