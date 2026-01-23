"""Abstract base classes to simulate `System`s forward in time."""

from collections.abc import Callable
from functools import cached_property

from jax import lax
from jax import numpy as jnp

from ..system.base import BaseSystem, System_ModelKnown

jndarray = jnp.ndarray


class BaseSolver:
    """Base class for solving true and data assimilated systems.

    Parameters
    ----------
    system
        An instance of `BaseSystem` to simulate forward in time.

    Methods
    -------
    solve_true
        Given a `System_ModelKnown`, simulate the true system forward in time.
    solve
        Simulate true and data assimilated systems forward in time
        simultaneously.

    Abstract Methods
    ----------------
    These must be overridden by subclasses.

    _step_factory
        Should return `step` functions to be used in `solve` methods.
    """

    def __init__(self, system: BaseSystem):
        self._system = system

    def _init_solve(
        self,
        state0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Initialize the arrays in which to store values computed in `solve`.

        Parameters
        ----------
        state0
            Initial value of system state (true or data assimilated)
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        state0
            Array initialized with inf with the shape to hold N steps of the
            system state
            shape (N, *state0.shape)
        tls
            The time linspace
        """
        num_steps = self.compute_num_steps(t0, tf, dt)

        # `arange` doesn't like floating point values.
        tls = t0 + jnp.arange(num_steps) * dt

        # Store the solution at every step.
        state = jnp.full(
            (num_steps, state0.shape[-1]), jnp.inf, dtype=state0.dtype
        )

        # Set initial state.
        state = state.at[: len(state0)].set(state0)

        return state, tls

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Solve the true system from `t0` to (approximately) `tf` with steps of
        size `dt`.

        Parameters
        ----------
        true0
            Initial state of true system
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        true
            True states
        tls
            Array of time points
        """
        raise NotImplementedError()

    def solve(
        self,
        true0: jndarray,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray]:
        """Solve the true and data assimilated systems together from `t0` to
        (approximately) `tf` with steps of size `dt`.

        Parameters
        ----------
        true0
            Initial state of true system
        assimilated0
            Initial state of data assimilated system
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        true
            True states
        assimilated
            Data assimilated states
        tls
            Array of time points
        """
        raise NotImplementedError()

    @staticmethod
    def compute_num_steps(t0: float, tf: float, dt: float) -> int:
        """Compute the number of time steps used to integrate over an interval.

        Parameters
        ----------
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        num_steps
            Number of steps used to integrate from `t0` to `tf` with steps of
            size `dt`
        """
        return round((tf - t0) / dt) + 1

    # The following attribute is read-only.
    system = property(lambda self: self._system)


class MultistageSolver(BaseSolver):
    """Abstract base class for non-multistep solvers (e.g., multistage solvers
    such as 4th-order Runge–Kutta).

    These solvers require that the system be of type `System_ModelKnown`, as
    `solve_true` requires this (of course) and multistage methods seem to
    require knowledge of the true model when nudging.
    """

    def __init__(self, system: System_ModelKnown):
        assert isinstance(system, System_ModelKnown), (
            "`system` must be of type `System_ModelKnown`"
        )

        super().__init__(system)

        self._step_true, self._step = self._step_factory()

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        if true0.ndim == 1:
            true0 = jnp.expand_dims(true0, 0)

        true, tls = self._init_solve(true0, t0, tf, dt)

        true, _ = lax.fori_loop(1, len(true), self._step_true, (true, (dt,)))

        return true, tls

    def solve(
        self,
        true0: jndarray,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray]:
        if true0.ndim == 1:
            true0 = jnp.expand_dims(true0, 0)
        if assimilated0.ndim == 1:
            assimilated0 = jnp.expand_dims(assimilated0, 0)

        true, tls = self._init_solve(true0, t0, tf, dt)
        assimilated, _ = self._init_solve(assimilated0, t0, tf, dt)

        (true, assimilated), _ = lax.fori_loop(
            1,
            len(true),
            self._step,
            ((true, assimilated), (dt, self.system.cs)),
        )

        return true, assimilated, tls

    def _step_factory(self) -> tuple[Callable, Callable]:
        """Define the `step` functions to be used in `solve`."""

        def step_true(i, vals):
            """Given the current state of the true system, compute the next
            state using `self.system.f_true`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.f_true`) will maintain the
            value used when `step` is first called (and thus compiled).
            """
            raise NotImplementedError()

        def step(i, vals):
            """Given the current state of the true and data assimilated systems
            and the estimated parameters for the data assimilated system,
            compute the next state using `self.system.f_true` and
            `self.system.f_assimilated`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.f_true`) will maintain the
            value used when `step` is first called (and thus compiled).
            """
            raise NotImplementedError()

        return step_true, step


class SinglestepSolver(BaseSolver):
    """Abstract base class for single-step solvers (e.g., forward Euler or
    backward Euler).

    See documentation for `BaseSolver`.

    Methods
    -------
    solve_assimilated
        Solve data assimilated system forward in time using observations of the
        true system state.
    """

    def __init__(self, system: BaseSystem):
        super().__init__(system)

        self._step_true, self._step_assimilated = self._step_factory()

    def solve_true(
        self, true0: jndarray, t0: float, tf: float, dt: float
    ) -> tuple[jndarray, jndarray]:
        assert isinstance(self.system, System_ModelKnown), (
            "`system` must be of type `System_ModelKnown`"
        )

        if true0.ndim == 1:
            true0 = jnp.expand_dims(true0, 0)

        true, tls = self._init_solve(true0, t0, tf, dt)

        true, _ = lax.fori_loop(1, len(true), self._step_true, (true, (dt,)))

        return true, tls

    def solve_assimilated(
        self,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
        ensure_optimized: bool = True,
    ) -> tuple[jndarray, jndarray]:
        """See documentation for `BaseSolver`.

        Parameters
        ----------
        assimilated0
            Initial state of data assimilated system
        true_observed
            Observed true states
        ensure_optimized
            If True, check whether `true_observed` is the exact length for the
            number of integration steps, raising a ValueError if `true_observed`
            contains too many states. See Notes section.

        Notes
        -----
        For optimal performance the exact number of observed true states
        required for the integration interval and step size should be passed to
        `true_observed`. It seems performance of jit-compiling is improved when
        at least one of the following conditions are met, but especially both:
            1. arrays from which slices are taken are the same size; and
            2. slices themselves are the same size.
        Passing the exact number of observed true states helps this code meet
        the first condition. This code meets the second condition when passing
        arrays to the jit-compiled `step` functions used in time integration
        solvers.

        Returns
        -------
        assimilated
            Data assimilated states
        tls
            Array of time points
        """
        if assimilated0.ndim == 1:
            assimilated0 = jnp.expand_dims(assimilated0, 0)

        assimilated, tls = self._init_solve(assimilated0, t0, tf, dt)

        if len(true_observed) < len(assimilated):
            raise IndexError("too few `true_observed` states given")
        if ensure_optimized:
            if len(true_observed) > len(assimilated):
                raise ValueError(
                    "too many `true_observed` states given; either pass"
                    " `ensure_optimized = False` or pass the exact number"
                    " of `true_observed` states for the time interval"
                )

        assimilated, _ = lax.fori_loop(
            1,
            len(assimilated),
            self._step_assimilated,
            (
                assimilated,
                (dt, self.system.cs, true_observed[: len(assimilated)]),
            ),
        )

        return assimilated, tls

    def solve(
        self,
        true0: jndarray,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray]:
        true, tls = self.solve_true(true0, t0, tf, dt)
        assimilated, _ = self.solve_assimilated(
            assimilated0,
            t0,
            tf,
            dt,
            true[:, self.system.true_observed_mask],
        )

        return true, assimilated, tls

    def _step_factory(self) -> tuple[Callable, Callable]:
        """Define the `step` functions to be used in `solve`."""

        def step_true(i, vals):
            """Given the current state of the true system, compute the next
            state using `self.system.f_true`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.f_true`) will maintain the
            value used when `step` is first called (and thus compiled).
            """
            raise NotImplementedError()

        def step_assimilated(i, vals):
            """Given the current state of the data assimilated system, its
            estimated parameters, and the observed portion of the true state,
            compute the next state using `self.system.f_assimilated`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.f_assimilated`) will
            maintain the value used when `step` is first called (and thus
            compiled).
            """
            raise NotImplementedError()

        return step_true, step_assimilated


class MultistepSolver(BaseSolver):
    """Abstract base class for multistep solvers (e.g., two-step
    Adams–Bashforth).

    See documentation for `BaseSolver`.

    Methods
    -------
    solve_assimilated
        Solve data assimilated system forward in time using observations of the
        true system state.

    Attributes
    ----------
    k
        Number of steps used in solver
        `_k` must be defined by subclasses (accessed through `k` defined in this
        class as a property).

    Properties
    ----------
    uses_multistage
        True if this solver instance uses a `MultistageSolver` at any point
    """

    def __init__(self, system: BaseSystem, pre_multistep_solver: BaseSolver):
        """

        Parameters
        ----------
        pre_multistep_solver
            An instantiated `BaseSolver` to use until enough steps have been
            taken to use the multistep solver
        """
        super().__init__(system)

        self._step_true, self._step_assimilated = self._step_factory()

        self._pre_multistep_solver = pre_multistep_solver

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "_k") or cls._k < 2:
            raise TypeError(
                f"{cls.__name__} must define class attribute '_k' >= 2;"
                " otherwise use `SinglestepSolver"
            )

    @cached_property
    def uses_multistage(self) -> bool:
        if isinstance(self._pre_multistep_solver, MultistageSolver):
            return True
        elif isinstance(self._pre_multistep_solver, MultistepSolver):
            return self._pre_multistep_solver.uses_multistage
        else:
            return False

    def solve_true(
        self, true0: jndarray, t0: float, tf: float, dt: float
    ) -> tuple[jndarray, jndarray]:
        """See documentation for `BaseSolver`.

        Parameters
        ----------
        true0
            Initial state(s) of true system
        """
        assert isinstance(self.system, System_ModelKnown), (
            "`system` must be of type `System_ModelKnown`"
        )

        if true0.ndim == 1:
            true0 = jnp.expand_dims(true0, 0)

        # Don't have enough steps to use this solver, so use
        # self._pre_multistep_solver to start.
        len0 = len(true0)
        if len0 < self.k:
            true, tls = self._init_solve(true0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # The time span is [t0, t0 + dt, ..., t0 + dt * (k-1)],
            # for a total of k steps.
            true0, _ = self._pre_multistep_solver.solve_true(
                true0, t0, t0 + dt * (self.k - 1), dt
            )

            true = true.at[len0 : len0 + self.k].set(true0)
        else:
            true, tls = self._init_solve(true0, t0 - dt * (self.k - 1), tf, dt)

        true, _ = lax.fori_loop(
            self.k, len(true), self._step_true, (true, (dt,))
        )

        return true, tls

    def solve_assimilated(
        self,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
        ensure_optimized: bool = True,
    ) -> tuple[jndarray, jndarray]:
        """See documentation for `BaseSolver`.

        Parameters
        ----------
        assimilated0
            Initial state(s) of data assimilated system
        true_observed
            Observed true states

            First entries should align with `assimilated0`.
        ensure_optimized
            If True, check whether `true_observed` is the exact length for the
            number of integration steps, raising a ValueError if `true_observed`
            contains too many states. See Notes section.

        Notes
        -----
        For optimal performance the exact number of observed true states
        required for the integration interval and step size should be passed to
        `true_observed`. It seems performance of jit-compiling is improved when
        at least one of the following conditions are met, but especially both:
            1. arrays from which slices are taken are the same size; and
            2. slices themselves are the same size.
        Passing the exact number of observed true states helps this code meet
        the first condition. This code meets the second condition when passing
        arrays to the jit-compiled `step` functions used in time integration
        solvers.

        Returns
        -------
        assimilated
            Data assimilated states
        tls
            Array of time points
        """
        if assimilated0.ndim == 1:
            assimilated0 = jnp.expand_dims(assimilated0, 0)

        # Don't have enough steps to use this solver, so use
        # self._pre_multistep_solver to start.
        len0 = len(assimilated0)
        if len0 < self.k:
            assimilated, tls = self._init_solve(assimilated0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # The time span is [t0, t0 + dt, ..., t0 + dt * (k-1)],
            # for a total of k steps.
            assimilated0, _ = self._pre_multistep_solver.solve_assimilated(
                assimilated0,
                t0,
                t0 + dt * (self.k - 1),
                dt,
                true_observed[: self.k],
            )

            assimilated = assimilated.at[len0 : len0 + self.k].set(assimilated0)
        else:
            assimilated, tls = self._init_solve(
                assimilated0, t0 - dt * (self.k - 1), tf, dt
            )

        if len(true_observed) < len(assimilated):
            raise IndexError("too few `true_observed` states given")
        if ensure_optimized:
            if len(true_observed) > len(assimilated):
                raise ValueError(
                    "too many `true_observed` states given; either pass"
                    " `ensure_optimized = False` or pass the exact number"
                    " of `true_observed` states for the time interval"
                )

        assimilated, _ = lax.fori_loop(
            self.k,
            len(assimilated),
            self._step_assimilated,
            (
                assimilated,
                (dt, self.system.cs, true_observed[: len(assimilated)]),
            ),
        )

        return assimilated, tls

    def solve(
        self,
        true0: jndarray,
        assimilated0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray]:
        len0 = len(assimilated0)
        if len0 < self.k and isinstance(
            self._pre_multistep_solver, MultistageSolver
        ):
            # Need k-1 previous steps to use k-step solver.
            # The time span is [t0, t0 + dt, ..., t0 + dt * (k-1)],
            # for a total of k steps.
            true0, assimilated0, _ = self._pre_multistep_solver.solve(
                true0, assimilated0, t0, t0 + dt * (self.k - 1), dt
            )

            t0 = t0 + dt * (self.k - 1)

        true, tls = self.solve_true(true0, t0, tf, dt)
        assimilated, _ = self.solve_assimilated(
            assimilated0, t0, tf, dt, true[:, self.system.true_observed_mask]
        )

        return true, assimilated, tls

    def _step_factory(self) -> tuple[Callable, Callable]:
        """Define the `step` functions to be used in `solve`."""

        def step_true(i, vals):
            """Given the current state of the true system, compute the next
            state using `self.system.f_true`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.f_true`) will maintain the
            value used when `step` is first called (and thus compiled).
            """
            raise NotImplementedError()

        def step_assimilated(i, vals):
            """Given the current state of the data assimilated system, its
            estimated parameters, and the observed portion of the true state,
            compute the next state using `self.system.f_assimilated`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.f_assimilated`) will
            maintain the value used when `step` is first called (and thus
            compiled).
            """
            raise NotImplementedError()

        return step_true, step_assimilated

    # The following attribute is read-only.
    k = property(lambda self: self._k)
