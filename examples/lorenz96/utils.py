from collections.abc import Callable

import numpy as np
from jax import numpy as jnp

from otf import optim
from otf.time_integration import base as ti_base
from otf.optim import base as optim_base
from otf.system.base import BaseSystem

jndarray = jnp.ndarray


def run_update(
    system: BaseSystem,
    solver: ti_base.SinglestepSolver | ti_base.MultistageSolver,
    dt: float,
    T0: float,
    Tf: float,
    t_relax: float,
    true0: jndarray,
    assimilated0: jndarray,
    true_shape: tuple[int, ...],
    assimilated_shape: tuple[int, ...],
    optimizer: Callable[[jndarray, jndarray], jndarray]
    | optim_base.BaseOptimizer
    | None = None,
):
    if optimizer is None:
        optimizer = optim.LevenbergMarquardt(system)

    t0 = T0
    tf = t0 + t_relax

    cs = [system.cs]

    u_errors = []

    record_v_errors = true_shape[1] == assimilated_shape[1]
    v_errors = [] if record_v_errors else None

    while tf <= Tf:
        true, assimilated, tls = solver.solve(true0, assimilated0, t0, tf, dt)

        true0, assimilated0 = true[-1], assimilated[-1]

        # Update parameters
        system.cs = optimizer(true0[system.true_observed_mask], assimilated0)
        cs.append(system.cs)

        t0 = tls[-1]
        tf = t0 + t_relax

        # Relative error
        u_errors.append(
            np.linalg.norm(
                true.reshape((-1, *true_shape))[1:, :, 0]
                - assimilated.reshape((-1, *assimilated_shape))[1:, :, 0],
                axis=1,
            )
            / np.linalg.norm(true.reshape((-1, *true_shape))[1:, :, 0], axis=1)
        )
        if record_v_errors:
            v_errors.append(
                np.linalg.norm(
                    true.reshape((-1, *true_shape))[1:, :, 1:]
                    - assimilated.reshape((-1, *assimilated_shape))[1:, :, 1:],
                    axis=(1, 2),
                )
                / np.linalg.norm(
                    true.reshape((-1, *true_shape))[1:, :, 1:], axis=(1, 2)
                )
            )

    u_errors = np.concatenate(u_errors)
    if record_v_errors:
        v_errors = np.concatenate(v_errors)

    # Note `t0` is the actual final time of the simulation (Tf_actual).
    tls = np.linspace(T0, t0, len(u_errors) + 1)

    return jnp.stack(cs), u_errors, v_errors, tls
