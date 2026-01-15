from enum import Enum


class UpdateOption(Enum):
    """Enum for selecting parameter update methods.

    Options include:

    - last_state: Uses the last observed state for the update.
    - mean_state: Uses the mean of the observed states for the update.
    - mean_gradient: Uses the mean of the gradients for the update.
    - adjoint: Uses adjoint method for the update.
    """

    last_state = 0
    mean_state = 1
    mean_gradient = 2
    adjoint = 3
