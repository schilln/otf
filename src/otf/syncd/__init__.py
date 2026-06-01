"""Public API for the `syncd` subpackage.

This package contains utilities for running synchronized simulations and
performing periodic parameter updates. The primary helpers live in
`otf.syncd.utils`.
"""

from .utils import run_update

__all__ = ["run_update"]

