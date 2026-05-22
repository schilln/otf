"""Linear–nonlinear time-integration schemes.

This module implements a set of time-integration schemes that split a system
into a linear part (handled analytically or implicitly) and a nonlinear part
handled explicitly. Implementations follow Cox & Matthews (2002) where
applicable and include exponential time differencing (ETD) and several multistep
predictor/corrector hybrids.

References
----------
Cox, S. M. and Matthews, P. C. (2002). Exponential Time Differencing for Stiff
Systems. Journal of Computational Physics, 176(2), 430-455.
doi:10.1006/jcph.2002.6995
"""

from jax import numpy as jnp

from .base import MultistepSolver, SinglestepSolver

jndarray = jnp.ndarray


class ETD1(SinglestepSolver):
    """First-order exponential time-differencing (ETD1) single-step solver.

    Uses an analytic integration of the linear part and a first-order update for
    the nonlinear contribution. Suitable for problems where the linear operator
    can be diagonalized/represented elementwise.
    """

    def _step_factory(self):
        def step_true(i, vals):
            l, nl_fn = self.system.linear_true(), self.system.nonlinear_true_ode

            true, (dt,) = vals
            t = true[i - 1]

            nl = nl_fn(t)
            nz = l != 0

            safe_l = jnp.where(nz, l, 1)
            out = jnp.expm1(l * dt) / safe_l
            out = jnp.where(nz, out, dt)

            true = true.at[i].set(t * jnp.exp(l * dt) + nl * out)

            return true, (dt,)

        def step_assimilated(i, vals):
            assimilated, (dt, cs, true_observed) = vals

            l, nl_fn = (
                self.system.linear_assimilated(cs),
                self.system.nudged_nonlinear_assimilated_ode,
            )
            t = true_observed[i - 1]
            a = assimilated[i - 1]

            nl = nl_fn(cs, t, a)
            nz = l != 0

            safe_l = jnp.where(nz, l, 1)
            out = jnp.expm1(l * dt) / safe_l
            out = jnp.where(nz, out, dt)

            assimilated = assimilated.at[i].set(t * jnp.exp(l * dt) + nl * out)

            return assimilated, (dt, cs, true_observed)

        return step_true, step_assimilated


class ETD2(MultistepSolver):
    """Second-order exponential time-differencing multistep solver (ETD2).

    A two-step ETD method that combines exact linear evolution with a
    second-order treatment of nonlinear terms.
    """

    _k = 2

    def _step_factory(self):
        def step_true(i, vals):
            l, nl_fn = self.system.linear_true(), self.system.nonlinear_true_ode
            nz = l != 0

            true, (dt,) = vals
            t2 = true[i - 2]
            t1 = true[i - 1]

            nl2 = nl_fn(t2)
            nl1 = nl_fn(t1)

            safe_l = jnp.where(nz, dt * l**2, 1)

            out1 = ((1 + dt * l) * jnp.exp(l * dt) - 1 - 2 * dt * l) / safe_l
            out1 = jnp.where(nz, out1, 3 * dt / 2)

            out2 = (-jnp.expm1(l * dt) + dt * l) / safe_l
            out2 = jnp.where(nz, out2, -dt / 2)

            true = true.at[i].set(
                t1 * jnp.exp(l * dt) + nl1 * out1 + nl2 * out2
            )

            return true, (dt,)

        def step_assimilated(i, vals):
            assimilated, (dt, cs, true_observed) = vals

            l, nl_fn = (
                self.system.linear_assimilated(cs),
                self.system.nudged_nonlinear_assimilated_ode,
            )
            nz = l != 0
            safe_l = jnp.where(nz, dt * l**2, 1)

            t2 = true_observed[i - 2]
            t1 = true_observed[i - 1]
            a2 = assimilated[i - 2]
            a1 = assimilated[i - 1]

            nl2 = nl_fn(cs, t2, a2)
            nl1 = nl_fn(cs, t1, a1)

            out1 = ((1 + dt * l) * jnp.exp(l * dt) - 1 - 2 * dt * l) / safe_l
            out1 = jnp.where(nz, out1, 3 * dt / 2)

            out2 = (-jnp.expm1(l * dt) + dt * l) / safe_l
            out2 = jnp.where(nz, out2, -dt / 2)

            assimilated = assimilated.at[i].set(
                t1 * jnp.exp(l * dt) + nl1 * out1 + nl2 * out2
            )

            return assimilated, (dt, cs, true_observed)

        return step_true, step_assimilated


class AB2AM2(MultistepSolver):
    """Adams–Bashforth 2 predictor with Adams–Moulton 2 corrector (AB2-AM2).

    A two-step predictor/corrector hybrid that treats the linear part implicitly
    (AM2 corrector) and the nonlinear part explicitly (AB2 predictor).
    """

    _k = 2

    def _step_factory(self):
        def step_true(i, vals):
            l, nl_fn = self.system.linear_true(), self.system.nonlinear_true_ode

            true, (dt,) = vals
            t2 = true[i - 2]
            t1 = true[i - 1]

            nl2 = nl_fn(t2)
            nl1 = nl_fn(t1)

            true = true.at[i].set(
                ((1 + dt / 2 * l) * t1 + 3 * dt / 2 * nl1 - dt / 2 * nl2)
                / (1 - dt / 2 * l)
            )

            return true, (dt,)

        def step_assimilated(i, vals):
            assimilated, (dt, cs, true_observed) = vals

            l, nl_fn = (
                self.system.linear_assimilated(cs),
                self.system.nudged_nonlinear_assimilated_ode,
            )
            t2 = true_observed[i - 2]
            t1 = true_observed[i - 1]
            a2 = assimilated[i - 2]
            a1 = assimilated[i - 1]

            nl2 = nl_fn(cs, t2, a2)
            nl1 = nl_fn(cs, t1, a1)

            assimilated = assimilated.at[i].set(
                ((1 + dt / 2 * l) * a1 + 3 * dt / 2 * nl1 - dt / 2 * nl2)
                / (1 - dt / 2 * l)
            )

            return assimilated, (dt, cs, true_observed)

        return step_true, step_assimilated


class AB2BD2(MultistepSolver):
    """Adams–Bashforth 2 predictor combined with a BDF/Backward-Differentiation
    style second-order corrector (AB2-BD2).

    A two-step hybrid that uses an explicit AB2 prediction and a stabilized
    second-order corrector for stiff linear components.
    """

    _k = 2

    def _step_factory(self):
        def step_true(i, vals):
            l, nl_fn = self.system.linear_true(), self.system.nonlinear_true_ode

            true, (dt,) = vals
            t2 = true[i - 2]
            t1 = true[i - 1]

            nl2 = nl_fn(t2)
            nl1 = nl_fn(t1)

            true = true.at[i].set(
                (4 * t1 - t2 + 4 * dt * nl1 - 2 * dt * nl2) / (3 - 2 * dt * l)
            )

            return true, (dt,)

        def step_assimilated(i, vals):
            assimilated, (dt, cs, true_observed) = vals

            l, nl_fn = (
                self.system.linear_assimilated(cs),
                self.system.nudged_nonlinear_assimilated_ode,
            )
            t2 = true_observed[i - 2]
            t1 = true_observed[i - 1]
            a2 = assimilated[i - 2]
            a1 = assimilated[i - 1]

            nl2 = nl_fn(cs, t2, a2)
            nl1 = nl_fn(cs, t1, a1)

            assimilated = assimilated.at[i].set(
                (4 * a1 - a2 + 4 * dt * nl1 - 2 * dt * nl2) / (3 - 2 * dt * l)
            )

            return assimilated, (dt, cs, true_observed)

        return step_true, step_assimilated
