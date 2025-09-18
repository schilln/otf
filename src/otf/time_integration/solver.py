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
