"""Public solver implementations and convenience exports for time integration.

This package exposes a small set of concrete time-integration algorithms used by
`otf` including single-step, multistep, multistage methods and a
`scipy.integrate.solve_ivp` adapter. Import the classes below for common
integration workflows.
"""

from .linear_nonlinear import (
    AB2AM2,
    AB2BD2,
    ETD1,
    ETD2,
)
from .solver import (
    RK4,
    ForwardEuler,
    FourStepAdamsBashforth,
    SolveIvp,
    TwoStepAdamsBashforth,
)

__all__ = [
    "AB2AM2",
    "AB2BD2",
    "ETD1",
    "ETD2",
    "RK4",
    "ForwardEuler",
    "FourStepAdamsBashforth",
    "SolveIvp",
    "TwoStepAdamsBashforth",
]
