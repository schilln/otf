from jax import numpy as jnp

from ..system import BaseSystem
from .base import MultistageSolver, MultistepSolver, BaseSolver

jndarray = jnp.ndarray


class ForwardEuler(MultistepSolver):
    def __init__(self, system: BaseSystem):
        """Forward Euler solver.

        See documentation of `base_solver.SinglestepSolver`.
        """
        super().__init__(system, None, 1)

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        start_with_multistep: bool = True,
    ) -> tuple[jndarray, jndarray]:
        # Note: `start_with_multistep` is ignored but kept for consistency with
        # multistep solvers.
        return super().solve_true(true0, t0, tf, dt, True)

    def solve_assimilated(
        self,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
        start_with_multistep: bool = True,
    ):
        # Note: `start_with_multistep` is ignored but kept for consistency with
        # multistep solvers.
        return super().solve_assimilated(
            assimilated0, t0, tf, dt, true_observed, True
        )

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
    def __init__(self, system: BaseSystem, pre_multistep_solver: BaseSolver):
        """Two-step Adams–Bashforth solver.

        See documentation of `base_solver.MultistepSolver`.

        See https://en.wikipedia.org/wiki/Linear_multistep_method#Two-step_Adams%E2%80%93Bashforth
        """

        super().__init__(system, pre_multistep_solver, 2)

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
