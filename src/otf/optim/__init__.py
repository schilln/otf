"""Optimization utilities and implementations used by the package.

This package exposes optimizers, learning-rate schedulers and gradient computers
used for parameter estimation when assimilating data into `BaseSystem`
instances.
"""

from . import gradient
from .base import (
    OptimizerChain,
    PartialOptimizer,
    Regularizer,
    pruned_factory,
)
from .gradient import (
    AdjointGradient,
    SensitivityGradient,
)
from .lr_scheduler import (
    DummyLRScheduler,
    ExponentialLR,
    MultiStepLR,
)
from .optimizer import (
    DummyOptimizer,
    GradientDescent,
    LevenbergMarquardt,
    OptaxWrapper,
    WeightedLevenbergMarquardt,
)

__all__ = [
    "AdjointGradient",
    "DummyLRScheduler",
    "DummyOptimizer",
    "ExponentialLR",
    "gradient",
    "GradientDescent",
    "LevenbergMarquardt",
    "MultiStepLR",
    "OptaxWrapper",
    "OptimizerChain",
    "PartialOptimizer",
    "pruned_factory",
    "Regularizer",
    "SensitivityGradient",
    "WeightedLevenbergMarquardt",
]
