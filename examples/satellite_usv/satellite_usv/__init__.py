"""Minimal satellite-to-USV coverage simulation."""

from .model import (
    GridSpec,
    SimulationConfig,
    SimulationHistory,
    USV,
    build_initial_satellite_field,
    build_satellite_consistent_truth,
    coarse_to_fine,
    fine_to_coarse,
    run_simulation,
)

__all__ = [
    "GridSpec",
    "SimulationConfig",
    "SimulationHistory",
    "USV",
    "build_initial_satellite_field",
    "build_satellite_consistent_truth",
    "coarse_to_fine",
    "fine_to_coarse",
    "run_simulation",
]
