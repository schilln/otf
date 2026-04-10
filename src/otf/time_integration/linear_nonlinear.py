from jax import numpy as jnp

from .base import SinglestepSolver

jndarray = jnp.ndarray


class ETD1(SinglestepSolver):
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
            pass

        return step_true, step_assimilated
