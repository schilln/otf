"""Learning-rate schedulers for optimizers.

Provides a small set of scheduler helpers that adjust an optimizer's
`learning_rate` attribute over time. Implementations are lightweight and
intended to be used with `BaseOptimizer` instances in the `optim` package.
"""

from collections import Counter

from .base import BaseOptimizer


class LRScheduler:
    """Abstract base for learning-rate schedulers that modify an optimizer's
    `learning_rate` attribute.
    """

    def __init__(self, optimizer: BaseOptimizer):
        """Create a scheduler bound to `optimizer`.

        Parameters
        ----------
        optimizer
            `BaseOptimizer` instance whose `learning_rate` attribute will be
            adjusted.
        """
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError


class DummyLRScheduler(LRScheduler):
    """No-op scheduler for testing and compatibility."""

    def __init__(self, *args, **kwargs):
        """Initialize a dummy scheduler (accepts arbitrary args).

        This scheduler performs no action when `step` is called.
        """
        pass

    def step(self):
        pass


class ExponentialLR(LRScheduler):
    """Multiply an optimizer's learning rate by a constant factor on each
    `step()` call.
    """

    def __init__(self, optimizer: BaseOptimizer, gamma: float = 0.99):
        """Initialize the exponential scheduler.

        Parameters
        ----------
        optimizer
            An instance of `BaseOptimizer` with a `learning_rate` attribute.
        gamma
            Factor to multiply the learning rate by on each `step`.
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.optimizer.learning_rate *= self.gamma


class MultiStepLR(LRScheduler):
    """Reduce learning rate at specified step milestones."""

    def __init__(
        self,
        optimizer: BaseOptimizer,
        milestones: list[int] | tuple[int],
        gamma: float = 0.5,
    ):
        """Initialize the multi-step scheduler.

        Parameters
        ----------
        optimizer
            An instance of `BaseOptimizer` with a `learning_rate` attribute.
        milestones
            For each milestone, update the learning rate after that many calls
            to `step`. Specifying the same milestone multiple times multiplies
            the learning rate repeatedly at that milestone.
        gamma
            Factor by which to multiply the learning rate at each milestone.
        """
        super().__init__(optimizer)
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.steps in self.milestones:
            self.optimizer.learning_rate *= (
                self.gamma ** self.milestones[self.steps]
            )
