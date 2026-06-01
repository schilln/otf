"""Asynchronous data-assimilation helpers.

This subpackage contains utilities for running asynchronous data-assimilation
workflows with `otf` systems. The primary helper exported here is `run_update`
(in `utils`) which runs a `BaseSystem`, performs parameter updates using an
optimizer, and returns parameter trajectories and error statistics.
"""

from .utils import run_update

__all__ = ["run_update"]
