"""Abstract base classes to simulate `base_system.System`s forward in time."""

from collections.abc import Callable

from jax import numpy as jnp, lax

from base_system import System

jndarray = jnp.ndarray


class Solver:
    def __init__(self, system: System):
        """Base class for solving true and nudged systems together.

        Parameters
        ----------
        system
            An instance of `base_system.System` to simulate forward in time.

        Methods
        -------
        solve
            Simulate `self.system` forward in time.

        Abstract Methods
        ----------------
        These must be overridden by subclasses.

        _step_factory
            A method that returns the `step` function to be used in `solve`.
        """
        self._system = system
        self._step = self._step_factory()

    def _step_factory(self) -> Callable:
        """Define the `step` function to be used in `solve`.

        See the docstring of the abstract `step` function defined in
        `base_solver.Solver`.
        """

        def step(i, vals):
            """Given the current state of the true and nudged systems and the
            estimated parameters for the nudged system, compute the next state
            using `self.system.f`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.ode`) will maintain the
            value used when `step` is first called (and thus compiled).
            """
            raise NotImplementedError()

        return step

    def _init_solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Initialize the arrays in which to store values computed in `solve`.

        Parameters
        ----------
        true0
            Initial value of true system state
        nudged0
            Initial value of nudged system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        true
            Array initialized with inf with the shape to hold N steps of the
            true state
            shape (N, *true0.shape)
        nudged
            Array initialized with inf with the shape to hold N steps of the
            nudged state
            shape (N, *nudged0.shape)
        tls
            The time linspace
        """
        # `arange` doesn't like floating point values.
        tls = t0 + jnp.arange(round((tf - t0) / dt)) * dt
        N = len(tls)

        # Store the solution at every step.
        true = jnp.full((N, *true0.shape), jnp.inf, dtype=true0.dtype)
        nudged = jnp.full((N, *nudged0.shape), jnp.inf, dtype=nudged0.dtype)

        # Set initial state.
        true = true.at[0].set(true0)
        nudged = nudged.at[0].set(nudged0)

        return true, nudged, tls

    def solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Simulate `self.system` from `t0` to `tf` using step size `dt`.

        Note for the following example implementation:
        The values passed to `step` in the `fori_loop` will depend on how `step`
        is defined.

        Example implementation
        ----------------------
        true, nudged, tls = self._init_solve(true0, nudged0, t0, tf, dt)

        (true, nudged), _ = lax.fori_loop(
            1, len(true), self.step, ((true, nudged), (dt, self.system.cs))
        )

        return true, nudged

        Parameters
        ----------
        true0
            Initial value of true system state
        nudged0
            Initial value of nudged system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        true, nudged
            The computed true and nudged states from `t0` to (approximately)
            `tf`, excluding the initial states `true0` and `nudged0`
        tls
            The time linspace
        """
        raise NotImplementedError()

    # The following attributes are read-only.
    system = property(lambda self: self._system)
    step = property(lambda self: self._step)


class SinglestepSolver(Solver):
    """Abstract base class for non-multistep solvers (e.g., multistage solvers
    such as 4th-order Runge–Kutta).

    See documentation of `Solver`.
    """

    def solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray, jndarray]:
        true, nudged, tls = self._init_solve(true0, nudged0, t0, tf, dt)

        (true, nudged), _ = lax.fori_loop(
            1, len(true), self.step, ((true, nudged), (dt, self.system.cs))
        )

        return true, nudged, tls


class MultistepSolver(Solver):
    def __init__(self, system: System, pre_multistep_solver: Solver, k: int):
        """Abstract base class for multistep solvers (e.g., two-step
        Adams–Bashforth).

        See documentation of `Solver`.

        Parameters
        ----------
        pre_multistep_solver
            An instantiated `Solver` to use until enough steps have been taken
            to use the multistep solver
        k
            The number of steps used in this multistep solver
        """
        super().__init__(system)
        self._k = k

        self._pre_multistep_solver = pre_multistep_solver

    def solve(
        self,
        true0: jndarray,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        start_with_multistep: bool = False,
    ) -> tuple[jndarray, jndarray, jndarray]:
        """See documentation of `Solver`.

        If `start_with_multistep` is True, then `true0` and `nudged0` should
        have shape (k, ...) where k is the number of steps used in the multistep
        solver, and the remaining dimensions are as usual (i.e., contain the
        state at one step).
        """
        # Don't have enough steps to use multistep solver, so use some other
        # solver to start.
        if not start_with_multistep:
            true, nudged, tls = self._init_solve(true0, nudged0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # Note upper bound is exclusive, so the span is really
            # [t0, t0 + dt, ..., t0 + dt * (k-1)], for a total of k steps.
            true0, nudged0, _ = self._pre_multistep_solver.solve(
                true0, nudged0, t0, t0 + dt * self.k, dt
            )

            true = true.at[1 : self.k].set(true0[1:])
            nudged = nudged.at[1 : self.k].set(nudged0[1:])

            (true, nudged), _ = lax.fori_loop(
                self.k,
                len(true),
                self.step,
                ((true, nudged), (dt, self.system.cs)),
            )

            return true, nudged, tls
        else:
            true, nudged, tls = self._init_solve(
                true0[0], nudged0[0], t0 - dt * (self.k - 1), tf, dt
            )
            true = true.at[1 : self.k].set(true0[1:])
            nudged = nudged.at[1 : self.k].set(nudged0[1:])

            (true, nudged), _ = lax.fori_loop(
                self.k,
                len(true),
                self.step,
                ((true, nudged), (dt, self.system.cs)),
            )

            return true, nudged, tls

    # The following attribute is read-only.
    k = property(lambda self: self._k)
