"""

Base Classes
------------
LRScheduler
    Abstract base class to implement learning rate scheduling

Classes Implementing Learning Rate Scheduling
---------------------------------------------
DummyLRScheduler
ExponentialLR
MultiStepLR
"""

from collections import Counter

from .base import BaseOptimizer


class LRScheduler:
    def __init__(self, optimizer: BaseOptimizer):
        """Given an `optimizer` with a `learning_rate` attribute, adjust its
        learning rate according to some algorithm.
        """
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError


class DummyLRScheduler(LRScheduler):
    def __init__(self, *args, **kwargs):
        """A dummy learning rate scheduler for testing with code that assumes
        use of a scheduler.
        """
        pass

    def step(self):
        pass


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer: BaseOptimizer, gamma: float = 0.99):
        """Multiply the optimizer's learning rate by a factor each time the
        method `step` is called.

        Parameters
        ----------
        optimizer
            An instance of `Optimizer` with a `learning_rate` attribute.
        gamma
            Multiply the learning rate of `optimizer` by `gamma` with every call
            to `step`.
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.optimizer.learning_rate *= self.gamma


class MultiStepLR(LRScheduler):
    def __init__(
        self,
        optimizer: BaseOptimizer,
        milestones: list[int] | tuple[int],
        gamma: float = 0.5,
    ):
        """At each given milestone (number of iterations), multiply the learning
        rate by a corresponding factor.

        Inspired by PyTorch's `MultiStepLR`

        Parameters
        ----------
        optimizer
            An instance of `Optimizer` with a `learning_rate` attribute.
        milestones
            For each milestone, update the learning rate after that many calls
            to `step`.
            Specifying the same milestone m times will result in
            multiplying the learning rate by `gamma` m times at that milestone.
        gamma
            Multiply the learning rate of `optimizer` by `gamma` upon reaching
            each milestone in `milestones`.
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
