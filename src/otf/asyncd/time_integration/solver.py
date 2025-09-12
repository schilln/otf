"""Concrete implementations of `base_solver.Solver`."""

from jax import numpy as jnp

from separate_base_system import System
from separate_base_solver import Solver, SinglestepSolver, MultistepSolver

jndarray = jnp.ndarray


class RK4(SinglestepSolver):
    """4th-order Runge–Kutta solver.

    See documentation of `base_solver.SinglestepSolver`.

    See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    """

    def _step_factory(self):
        def step_true(i, vals):
            f = self.system.f_true

            true, (dt,) = vals
            t = true[i - 1]

            k1t = f(t)
            k2t = f(t + dt * k1t / 2)
            k3t = f(t + dt * k2t / 2)
            k4t = f(t + dt * k3t)

            t = t.at[:].add((dt / 6) * (k1t + 2 * k2t + 2 * k3t + k4t))

            true = true.at[i].set(t)

            return true, (dt,)

        def step_nudged(i, vals):
            raise NotImplementedError("Not yet implemented for nudged system.")

        return step_true, step_nudged


class ForwardEuler(SinglestepSolver):
    def __init__(self, system: System):
        """Forward Euler solver.

        See documentation of `base_solver.SinglestepSolver`.
        """
        super().__init__(system)

    def _step_factory(self):
        def step_true(i, vals):
            f = self.system.f_true

            true, (dt,) = vals
            t = true[i - 1]

            t = t.at[:].add(dt * f(t))

            true = true.at[i].set(t)

            return true, (dt,)

        def step_nudged(i, vals):
            f = self.system.f_nudged

            nudged, (dt, cs, true_observed) = vals
            t = true_observed[i - 1]
            n = nudged[i - 1]

            n = n.at[:].add(dt * f(cs, t, n))

            nudged = nudged.at[i].set(n)

            return nudged, (dt, cs, true_observed)

        return step_true, step_nudged


class TwoStepAdamsBashforth(MultistepSolver):
    def __init__(self, system: System, pre_multistep_solver: Solver):
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

            t1 = t1.at[:].add(3 / 2 * dt * tmp1[0] - 1 / 2 * dt * tmp2[0])

            true = true.at[i].set(t1)

            return true, (dt,)

        def step_nudged(i, vals):
            f = self.system.f_nudged

            nudged, (dt, cs, true_observed) = vals
            t2, n2 = true_observed[i - 2], nudged[i - 2]
            t1, n1 = true_observed[i - 1], nudged[i - 1]

            tmp2 = f(cs, t2, n2)
            tmp1 = f(cs, t1, n1)

            n1 = n1.at[:].add(3 / 2 * dt * tmp1 - 1 / 2 * dt * tmp2)

            nudged = nudged.at[i].set(n1)

            return nudged, (dt, cs, true_observed)

        return step_true, step_nudged


class FourStepAdamsBashforth(MultistepSolver):
    def __init__(self, system: System, pre_multistep_solver: Solver):
        """Four-step Adams–Bashforth solver.

        See documentation of `base_solver.MultistepSolver`.

        https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods
        """

        super().__init__(system, pre_multistep_solver, 4)

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

        def step_nudged(i, vals):
            f = self.system.f_nudged

            nudged, (dt, cs, true_observed) = vals
            t4, n4 = true_observed[i - 4], nudged[i - 4]
            t3, n3 = true_observed[i - 3], nudged[i - 3]
            t2, n2 = true_observed[i - 2], nudged[i - 2]
            t1, n1 = true_observed[i - 1], nudged[i - 1]

            p4 = f(cs, t4, n4)
            p3 = f(cs, t3, n3)
            p2 = f(cs, t2, n2)
            p1 = f(cs, t1, n1)

            n1 = n1.at[:].add(dt / 24 * (55 * p1 - 59 * p2 + 37 * p3 - 9 * p4))

            nudged = nudged.at[i].set(n1)

            return nudged, (dt, cs, true_observed)

        return step_true, step_nudged
