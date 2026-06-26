#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .model import (
    GridSpec,
    USV,
    diffusion_laplacian,
    fine_to_coarse,
    source_field,
)

ControlPlanner = Literal["no_control", "random_control", "greedy_fine", "hierarchical"]


@dataclass
class ControlConfig:
    """Configuration for the environment-reduction control demo."""

    steps: int = 400
    seed: int = 7
    planner: ControlPlanner = "hierarchical"
    usv_count: int = 3
    dt: float = 0.2
    diffusion_rate: float = 0.02
    decay_rate: float = 0.0
    source_strength: float = 0.0
    control_gain: float = 0.55
    control_amount: float = 1.0
    target_scale: float = 0.82
    tracking_weight: float = 50.0
    control_weight: float = 0.02
    move_weight: float = 0.001
    stay_weight: float = 0.025
    target_move_weight: float = 0.002
    assigned_coarse_weight: float = 0.08
    local_reduction_weight: float = 20.0
    depleted_control_penalty: float = 0.02
    hotspot_fraction: float = 1.8


@dataclass
class ControlHistory:
    """Recorded arrays from the environment-reduction control demo."""

    truth: list[NDArray[np.float64]] = field(default_factory=list)
    coarse_output: list[NDArray[np.float64]] = field(default_factory=list)
    target_output: list[NDArray[np.float64]] = field(default_factory=list)
    residual: list[NDArray[np.float64]] = field(default_factory=list)
    control: list[NDArray[np.float64]] = field(default_factory=list)
    usv_positions: list[list[tuple[int, int]]] = field(default_factory=list)
    tracking_error: list[float] = field(default_factory=list)
    total_concentration: list[float] = field(default_factory=list)
    max_concentration: list[float] = field(default_factory=list)
    control_cost: list[float] = field(default_factory=list)


def run_control_simulation(
    config: ControlConfig | None = None,
    grid: GridSpec | None = None,
    initial_truth: NDArray[np.float64] | None = None,
) -> ControlHistory:
    """Run one environment-reduction control simulation."""
    config = config or ControlConfig()
    grid = grid or GridSpec(coarse_shape=(4, 4), fine_per_coarse=(4, 4))
    rng = np.random.default_rng(config.seed)

    truth = build_control_truth(grid) if initial_truth is None else initial_truth.copy()

    usvs = _initial_usvs(grid, config.usv_count)
    history = ControlHistory()

    initial_output = fine_to_coarse(grid, truth)
    target_output = build_upper_target(grid, initial_output, config)
    _record(
        history,
        grid,
        truth,
        initial_output,
        target_output,
        np.zeros(grid.fine_shape, dtype=np.float64),
        usvs,
    )

    for step in range(config.steps):
        current_output = fine_to_coarse(grid, truth)
        control, usvs = _select_control(grid, truth, target_output, usvs, config, rng)
        source = source_field(grid, step, config.source_strength)
        truth = environment_step(truth, control, source, config)
        coarse_output = fine_to_coarse(grid, truth)
        _record(history, grid, truth, coarse_output, target_output, control, usvs)

    return history


def run_control_comparison(
    config: ControlConfig | None = None,
    grid: GridSpec | None = None,
) -> dict[str, ControlHistory]:
    """Run all baseline planners from the same initial condition."""
    config = config or ControlConfig()
    grid = grid or GridSpec(coarse_shape=(4, 4), fine_per_coarse=(4, 4))
    initial_truth = build_control_truth(grid)

    histories: dict[str, ControlHistory] = {}
    for planner in ["no_control", "random_control", "greedy_fine", "hierarchical"]:
        planner_config = ControlConfig(**{**config.__dict__, "planner": planner})
        histories[planner] = run_control_simulation(planner_config, grid, initial_truth)
    return histories


def build_upper_target(
    grid: GridSpec,
    initial_coarse_output: NDArray[np.float64],
    config: ControlConfig,
) -> NDArray[np.float64]:
    """Upper target: keep the upper-right coarse region high and lower the rest."""
    coarse_rows, coarse_cols = grid.coarse_shape
    row_axis = np.linspace(-1.0, 1.0, coarse_rows)
    col_axis = np.linspace(-1.0, 1.0, coarse_cols)
    rr, cc = np.meshgrid(row_axis, col_axis, indexing="ij")
    target_shape = 0.16
    target_shape += 0.78 * np.exp(-((rr - 0.70) ** 2 + (cc - 0.70) ** 2) / 0.45)
    target_shape += 0.08 * (cc + 1.0)
    target_shape = target_shape / max(float(np.max(target_shape)), 1e-9)
    return config.target_scale * float(np.max(initial_coarse_output)) * target_shape


def build_control_satellite_field(grid: GridSpec) -> NDArray[np.float64]:
    """Create the coarse output that the upper planner would see at k=0."""
    return fine_to_coarse(grid, build_control_truth(grid))


def build_control_truth(grid: GridSpec) -> NDArray[np.float64]:
    """Create a broad smooth field whose right half is initially high."""
    rows, cols = grid.fine_shape
    row_axis = np.linspace(-1.0, 1.0, rows)
    col_axis = np.linspace(-1.0, 1.0, cols)
    rr, cc = np.meshgrid(row_axis, col_axis, indexing="ij")
    right_weight = 0.5 * (np.tanh(3.0 * cc) + 1.0)
    field = 0.12
    field += 0.62 * right_weight
    field += 0.45 * np.exp(-((rr - 0.05) ** 2 + (cc - 0.45) ** 2) / 1.35)
    field += 0.08 * (rr + 1.0)
    return np.asarray(field, dtype=np.float64)


def environment_step(
    field: NDArray[np.float64],
    control: NDArray[np.float64],
    source: NDArray[np.float64],
    config: ControlConfig,
) -> NDArray[np.float64]:
    """Advance x_{k+1} = x_k + dt(DLx - lambda x + q - beta u)."""
    next_field = field + config.dt * (
        config.diffusion_rate * diffusion_laplacian(field)
        - config.decay_rate * field
        + source
        - config.control_gain * control
    )
    return np.maximum(next_field, 0.0)


def _select_control(
    grid: GridSpec,
    truth: NDArray[np.float64],
    target_output: NDArray[np.float64],
    usvs: list[USV],
    config: ControlConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], list[USV]]:
    if config.planner == "no_control":
        return np.zeros(grid.fine_shape, dtype=np.float64), usvs

    control = np.zeros(grid.fine_shape, dtype=np.float64)
    reserved: set[tuple[int, int]] = set()
    reserved_coarse: set[tuple[int, int]] = set()
    planned_truth = truth.copy()
    next_usvs: list[USV] = []

    for usv in usvs:
        candidates = [
            candidate for candidate in grid.neighbors(usv.position, include_stay=True) if candidate not in reserved
        ]
        if not candidates:
            candidates = [usv.position]

        if config.planner == "random_control":
            selected = candidates[int(rng.integers(0, len(candidates)))]
        elif config.planner == "greedy_fine":
            selected = max(candidates, key=lambda candidate: truth[candidate])
        else:
            target_coarse = _select_control_target_coarse(grid, planned_truth, target_output, reserved_coarse)
            reserved_coarse.add(target_coarse)
            selected = min(
                candidates,
                key=lambda candidate: _hierarchical_candidate_cost(
                    grid,
                    planned_truth,
                    target_output,
                    usv.position,
                    candidate,
                    target_coarse,
                    config,
                ),
            )

        control[selected] += config.control_amount
        planned_truth = np.maximum(
            planned_truth - config.control_gain * config.dt * _point_kernel(grid, selected),
            0.0,
        )
        reserved.add(selected)
        next_usvs.append(USV(position=selected))

    return control, next_usvs


def _hierarchical_candidate_cost(
    grid: GridSpec,
    truth: NDArray[np.float64],
    target_output: NDArray[np.float64],
    current_position: tuple[int, int],
    candidate: tuple[int, int],
    target_coarse: tuple[int, int],
    config: ControlConfig,
) -> float:
    control = config.control_amount * _point_kernel(grid, candidate)
    predicted = environment_step(truth, control, np.zeros(grid.fine_shape, dtype=np.float64), config)
    predicted_output = fine_to_coarse(grid, predicted)
    current_output = fine_to_coarse(grid, truth)
    tracking_cost = config.tracking_weight * float(np.sum((predicted_output - target_output) ** 2))
    control_cost = config.control_weight * config.control_amount**2
    move_cost = config.move_weight * _move_cost(current_position, candidate)
    stay_cost = config.stay_weight if candidate == current_position else 0.0
    target_center = grid.coarse_center(target_coarse)
    target_move_cost = config.target_move_weight * _move_cost(candidate, target_center)
    candidate_coarse = grid.coarse_index_for_position(candidate)
    assigned_coarse_cost = 0.0 if candidate_coarse == target_coarse else config.assigned_coarse_weight
    coarse_need = max(float(current_output[candidate_coarse] - target_output[candidate_coarse]), 0.0)
    local_reduction = min(float(truth[candidate]), config.control_gain * config.dt * config.control_amount)
    local_reduction_reward = config.local_reduction_weight * coarse_need * local_reduction
    depleted_penalty = config.depleted_control_penalty if local_reduction <= 1e-9 else 0.0
    return (
        tracking_cost
        + control_cost
        + move_cost
        + stay_cost
        + target_move_cost
        + assigned_coarse_cost
        + depleted_penalty
        - local_reduction_reward
    )


def _select_control_target_coarse(
    grid: GridSpec,
    truth: NDArray[np.float64],
    target_output: NDArray[np.float64],
    reserved_coarse: set[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    residual = fine_to_coarse(grid, truth) - target_output
    ranked = np.argsort(residual.reshape(-1))[::-1]
    reserved_coarse = reserved_coarse or set()
    for coarse_index_flat in ranked:
        coarse_index = np.unravel_index(int(coarse_index_flat), grid.coarse_shape)
        if coarse_index not in reserved_coarse:
            return coarse_index
    coarse_index_flat = int(ranked[0])
    return np.unravel_index(coarse_index_flat, grid.coarse_shape)


def _point_kernel(grid: GridSpec, position: tuple[int, int]) -> NDArray[np.float64]:
    kernel = np.zeros(grid.fine_shape, dtype=np.float64)
    kernel[position] = 1.0
    return kernel


def _move_cost(current_position: tuple[int, int], next_position: tuple[int, int]) -> float:
    return float(abs(current_position[0] - next_position[0]) + abs(current_position[1] - next_position[1]))


def _record(
    history: ControlHistory,
    grid: GridSpec,
    truth: NDArray[np.float64],
    coarse_output: NDArray[np.float64],
    target_output: NDArray[np.float64],
    control: NDArray[np.float64],
    usvs: list[USV],
) -> None:
    residual = coarse_output - target_output
    history.truth.append(truth.copy())
    history.coarse_output.append(coarse_output.copy())
    history.target_output.append(target_output.copy())
    history.residual.append(residual.copy())
    history.control.append(control.copy())
    history.usv_positions.append([usv.position for usv in usvs])
    history.tracking_error.append(float(np.sqrt(np.sum(residual**2))))
    history.total_concentration.append(float(np.sum(truth)))
    history.max_concentration.append(float(np.max(truth)))
    history.control_cost.append(float(np.sum(control**2)))


def _initial_usvs(grid: GridSpec, usv_count: int) -> list[USV]:
    rows, cols = grid.fine_shape
    candidates = [(rows - 1, 0), (0, cols - 1), (rows - 1, cols - 1)]
    return [USV(position=grid.clip_position(candidates[index % len(candidates)])) for index in range(usv_count)]
