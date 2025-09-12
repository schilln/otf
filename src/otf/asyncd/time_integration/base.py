"""Abstract base classes to simulate `base_system.System`s forward in time."""

from collections.abc import Callable

from jax import numpy as jnp, lax

from separate_base_system import System

jndarray = jnp.ndarray


class Solver:
    def __init__(self, system: System):
        """Base class for solving true and nudged systems separately.

        Parameters
        ----------
        system
            An instance of `separate_base_system.System` to simulate forward in
            time.

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
        self._step_true, self._step_nudged = self._step_factory()

    def _step_factory(self) -> Callable:
        """Define the `step` function to be used in `solve`.

        See the docstring of the abstract `step_true` and `step_nudged`
        functions defined in `separate_base_solver.Solver`.
        """

        def step_true(i, vals):
            """Given the current state of the true system, compute the next
            state using `self.system.f_true`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.ode`) will maintain the
            value used when `step` is first called (and thus compiled).
            """
            raise NotImplementedError()

        def step_nudged(i, vals):
            """Given the current state of the nudged system and the
            estimated parameters for the nudged system, compute the next state
            using `self.system.f_nudged`.

            This function will be jitted, and in particular it will be used as
            the `body_fun` parameter of `lax.fori_loop`, so it must conform to
            that interface. See
            https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html

            Being jitted, only its parameters `i` and `vals` may be updated.
            Other values (such as accessing `system.ode`) will maintain the
            value used when `step` is first called (and thus compiled).
            """
            raise NotImplementedError()

        return step_true, step_nudged

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
            Initial value of system state (true or nudged)
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
        # `arange` doesn't like floating point values.
        tls = t0 + jnp.arange(round((tf - t0) / dt)) * dt
        N = len(tls)

        # Store the solution at every step.
        state = jnp.full((N, *state0.shape), jnp.inf)

        # Set initial state.
        state = state.at[0].set(state0)

        return state, tls

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        """Simulate true state of `self.system` from `t0` to `tf` using step
        size `dt`.

        Note for the following example implementation:
        The values passed to `step` in the `fori_loop` will depend on how `step`
        is defined.

        Example implementation
        ----------------------
        true, tls = self._init_solve(true0, t0, tf, dt)

        true, _ = lax.fori_loop(1, len(true), self.step, (true, (dt,)))

        return true

        Parameters
        ----------
        true0
            Initial value of true system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size

        Returns
        -------
        true
            The computed true states from `t0` to (approximately) `tf`,
            excluding the initial states `true0`
        tls
            The time linspace
        """
        raise NotImplementedError()

    def solve_nudged(
        self,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
    ) -> tuple[jndarray, jndarray]:
        """Simulate nudged state of `self.system` from `t0` to `tf` using step
        size `dt`.

        Note for the following example implementation:
        The values passed to `step` in the `fori_loop` will depend on how `step`
        is defined.

        Example implementation
        ----------------------
        nudged, tls = self._init_solve(nudged0, t0, tf, dt)

        nudged, _ = lax.fori_loop(
            1,
            len(nudged),
            self.step,
            (nudged, (dt, self.system.cs, true_observed)),
        )

        return nudged

        Parameters
        ----------
        nudged0
            Initial value of nudged system state
        t0, tf
            Initial and (approximate) final times over which to simulate
        dt
            Simulation step size
        true_observed
            Observed true states

        Returns
        -------
        nudged
            The computed nudged states from `t0` to (approximately)
            `tf`, excluding the initial states `nudged0`
        tls
            The time linspace
        """
        raise NotImplementedError()

    # The following attributes are read-only.
    system = property(lambda self: self._system)
    step_true = property(lambda self: self._step_true)
    step_nudged = property(lambda self: self._step_nudged)


class SinglestepSolver(Solver):
    """Abstract base class for non-multistep solvers (e.g., multistage solvers
    such as 4th-order Runge–Kutta)."""

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
    ) -> tuple[jndarray, jndarray]:
        true, tls = self._init_solve(true0, t0, tf, dt)

        true, _ = lax.fori_loop(1, len(true), self.step_true, (true, (dt,)))

        return true, tls

    def solve_nudged(
        self,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
    ) -> tuple[jndarray, jndarray]:
        nudged, tls = self._init_solve(nudged0, t0, tf, dt)

        nudged, _ = lax.fori_loop(
            1,
            len(nudged),
            self.step_nudged,
            (nudged, (dt, self.system.cs, true_observed)),
        )

        return nudged, tls


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

    def solve_true(
        self,
        true0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        start_with_multistep: bool = False,
    ) -> tuple[jndarray, jndarray]:
        """See documentation for `Solver`.

        The final entry of `true0` corresponds to `t0`, while preceding entries
        `true0[-2]`, `true0[-3]`, ... correspond to t0 - dt, t0 - 2*dt, ...

        If `start_with_multistep` is True, then `true0` should have shape
        (k, ...) where k is the number of steps used in the multistep solver,
        and the remaining dimensions are as usual (i.e., contain the state at
        one step).
        """
        # Don't have enough steps to use multistep solver, so use some other
        # solver to start.
        if not start_with_multistep:
            true, tls = self._init_solve(true0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # Note upper bound is exclusive, so the span is really
            # [t0, t0 + dt, ..., t0 + dt * (k-1)], for a total of k steps.
            true0, _ = self._pre_multistep_solver.solve_true(
                true0, t0, t0 + dt * self.k, dt
            )

            true = true.at[1 : self.k].set(true0[1:])

            true, _ = lax.fori_loop(
                self.k, len(true), self.step_true, (true, (dt,))
            )

            return true, tls
        else:
            true, tls = self._init_solve(
                true0[0], t0 - dt * (self.k - 1), tf, dt
            )
            true = true.at[1 : self.k].set(true0[1:])

            true, _ = lax.fori_loop(
                self.k, len(true), self.step_true, (true, (dt,))
            )

            return true, tls

    def solve_nudged(
        self,
        nudged0: jndarray,
        t0: float,
        tf: float,
        dt: float,
        true_observed: jndarray,
        start_with_multistep: bool = False,
    ) -> tuple[jndarray, jndarray]:
        """See documentation for `Solver`.

        The final entry of `nudged0` corresponds to `t0`, while preceding
        entries `nudged0[-2]`, `nudged0[-3]`, ... correspond to t0 - dt,
        t0 - 2*dt, ...

        If `start_with_multistep` is True, then `nudged0` should have shape
        (k, ...) where k is the number of steps used in the multistep solver,
        and the remaining dimensions are as usual (i.e., contain the state at
        one step).
        """
        # Don't have enough steps to use multistep solver, so use some other
        # solver to start.
        if not start_with_multistep:
            nudged, tls = self._init_solve(nudged0, t0, tf, dt)

            # Need k-1 previous steps to use k-step solver.
            # Note upper bound is exclusive, so the span is really
            # [t0, t0 + dt, ..., t0 + dt * (k-1)], for a total of k steps.
            nudged0, _ = self._pre_multistep_solver.solve_nudged(
                nudged0, t0, t0 + dt * self.k, dt, true_observed
            )

            nudged = nudged.at[1 : self.k].set(nudged0[1:])

            nudged, _ = lax.fori_loop(
                self.k,
                len(nudged),
                self.step_nudged,
                (nudged, (dt, self.system.cs, true_observed)),
            )

            return nudged, tls
        else:
            nudged, tls = self._init_solve(
                nudged0[0], t0 - dt * (self.k - 1), tf, dt
            )
            nudged = nudged.at[1 : self.k].set(nudged0[1:])

            true, _ = lax.fori_loop(
                self.k,
                len(nudged),
                self.step_nudged,
                (nudged, (dt, self.system.cs, true_observed)),
            )

            return true, tls

    # The following attribute is read-only.
    k = property(lambda self: self._k)
