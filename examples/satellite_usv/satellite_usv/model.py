#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

ControlMode = Literal["none", "ideal", "usv"]
PlannerMode = Literal["tracking", "satellite_survey", "greedy"]


@dataclass(frozen=True)
class GridSpec:
    """Two-level grid used by the satellite-USV toy problem."""

    coarse_shape: tuple[int, int] = (5, 5)
    fine_per_coarse: tuple[int, int] = (3, 3)

    @property
    def fine_shape(self) -> tuple[int, int]:
        return (
            self.coarse_shape[0] * self.fine_per_coarse[0],
            self.coarse_shape[1] * self.fine_per_coarse[1],
        )

    @property
    def coarse_count(self) -> int:
        return int(np.prod(self.coarse_shape))

    @property
    def fine_count(self) -> int:
        return int(np.prod(self.fine_shape))

    @property
    def cells_per_coarse(self) -> int:
        return int(np.prod(self.fine_per_coarse))

    def build_averaging_matrix(self) -> NDArray[np.float64]:
        """Return H where coarse satellite observations are x_bar = H x."""
        h = np.zeros((self.coarse_count, self.fine_count), dtype=np.float64)
        for coarse_index, fine_indices in enumerate(self.iter_coarse_blocks_flat()):
            h[coarse_index, list(fine_indices)] = 1.0 / self.cells_per_coarse
        return h

    def iter_coarse_blocks(self) -> Iterable[tuple[tuple[int, int], tuple[slice, slice]]]:
        rows_per_block, cols_per_block = self.fine_per_coarse
        for coarse_row in range(self.coarse_shape[0]):
            for coarse_col in range(self.coarse_shape[1]):
                row_slice = slice(coarse_row * rows_per_block, (coarse_row + 1) * rows_per_block)
                col_slice = slice(coarse_col * cols_per_block, (coarse_col + 1) * cols_per_block)
                yield (coarse_row, coarse_col), (row_slice, col_slice)

    def iter_coarse_blocks_flat(self) -> Iterable[tuple[int, ...]]:
        _, fine_cols = self.fine_shape
        for _, (row_slice, col_slice) in self.iter_coarse_blocks():
            indices: list[int] = []
            for row in range(row_slice.start, row_slice.stop):
                for col in range(col_slice.start, col_slice.stop):
                    indices.append(row * fine_cols + col)
            yield tuple(indices)

    def clip_position(self, position: tuple[int, int]) -> tuple[int, int]:
        rows, cols = self.fine_shape
        return (int(np.clip(position[0], 0, rows - 1)), int(np.clip(position[1], 0, cols - 1)))

    def neighbors(self, position: tuple[int, int], include_stay: bool = True) -> list[tuple[int, int]]:
        row, col = self.clip_position(position)
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        if include_stay:
            candidates.insert(0, (row, col))

        rows, cols = self.fine_shape
        return [(r, c) for r, c in candidates if 0 <= r < rows and 0 <= c < cols]

    def coarse_index_for_position(self, position: tuple[int, int]) -> tuple[int, int]:
        row, col = self.clip_position(position)
        return (row // self.fine_per_coarse[0], col // self.fine_per_coarse[1])

    def coarse_center(self, coarse_index: tuple[int, int]) -> tuple[int, int]:
        coarse_row, coarse_col = coarse_index
        return (
            coarse_row * self.fine_per_coarse[0] + self.fine_per_coarse[0] // 2,
            coarse_col * self.fine_per_coarse[1] + self.fine_per_coarse[1] // 2,
        )


@dataclass
class USV:
    """A grid-cell USV with local sensing and optional local control."""

    position: tuple[int, int]


@dataclass
class SimulationConfig:
    """Configuration for the small satellite-USV simulation."""

    steps: int = 40
    seed: int = 7
    control_mode: ControlMode = "none"
    planner_mode: PlannerMode = "tracking"
    usv_count: int = 2
    dt: float = 0.2
    diffusion_rate: float = 0.01
    decay_rate: float = 0.0
    source_strength: float = 0.0
    control_gain: float = 0.45
    control_limit: float = 0.7
    hotspot_fraction: float = 1.6
    process_noise_std: float = 0.0
    satellite_noise_std: float = 0.01
    usv_noise_std: float = 0.015
    usv_assimilation_gain: float = 0.85
    satellite_period: int = 5
    estimator_smoothing_gain: float = 0.02
    coarse_reference: float = 0.15
    survey_uncertainty_weight: float = 1.2
    survey_distance_weight: float = 0.03
    tracking_uncertainty_weight: float = 0.03
    tracking_distance_weight: float = 0.006
    tracking_target_penalty: float = 0.08
    observation_density_decay: float = 0.0


@dataclass
class SimulationHistory:
    """Recorded arrays from a simulation run."""

    truth: list[NDArray[np.float64]] = field(default_factory=list)
    estimate: list[NDArray[np.float64]] = field(default_factory=list)
    satellite: list[NDArray[np.float64]] = field(default_factory=list)
    control: list[NDArray[np.float64]] = field(default_factory=list)
    uncertainty: list[NDArray[np.float64]] = field(default_factory=list)
    observation_density: list[NDArray[np.float64]] = field(default_factory=list)
    y_ref: list[NDArray[np.float64]] = field(default_factory=list)
    y_output: list[NDArray[np.float64]] = field(default_factory=list)
    usv_positions: list[list[tuple[int, int]]] = field(default_factory=list)
    coarse_rmse: list[float] = field(default_factory=list)
    fine_rmse: list[float] = field(default_factory=list)
    weighted_rmse: list[float] = field(default_factory=list)
    tracking_error: list[float] = field(default_factory=list)
    total_concentration: list[float] = field(default_factory=list)
    observed_fraction: list[float] = field(default_factory=list)


def fine_to_coarse(grid: GridSpec, fine_field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average a fine field into the coarse satellite grid."""
    field_2d = np.asarray(fine_field, dtype=np.float64).reshape(grid.fine_shape)
    coarse = np.zeros(grid.coarse_shape, dtype=np.float64)
    for (coarse_row, coarse_col), block in grid.iter_coarse_blocks():
        coarse[coarse_row, coarse_col] = float(np.mean(field_2d[block]))
    return coarse


def coarse_to_fine(grid: GridSpec, coarse_field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expand each coarse satellite cell uniformly to its fine cells."""
    coarse_2d = np.asarray(coarse_field, dtype=np.float64).reshape(grid.coarse_shape)
    fine = np.zeros(grid.fine_shape, dtype=np.float64)
    for (coarse_row, coarse_col), block in grid.iter_coarse_blocks():
        fine[block] = coarse_2d[coarse_row, coarse_col]
    return fine


def build_initial_satellite_field(grid: GridSpec) -> NDArray[np.float64]:
    """Create a biased coarse satellite field with a clear high-importance region."""
    coarse_rows, coarse_cols = grid.coarse_shape
    row_axis = np.linspace(-1.0, 1.0, coarse_rows)
    col_axis = np.linspace(-1.0, 1.0, coarse_cols)
    rr, cc = np.meshgrid(row_axis, col_axis, indexing="ij")
    field = 0.10
    field += 0.78 * np.exp(-((rr - 0.45) ** 2 + (cc - 0.48) ** 2) / 0.18)
    field += 0.26 * np.exp(-((rr + 0.55) ** 2 + (cc + 0.15) ** 2) / 0.32)
    field += 0.06 * (rr + 1.0)
    return np.asarray(field, dtype=np.float64)


def build_satellite_consistent_truth(
    grid: GridSpec,
    satellite_field: NDArray[np.float64],
    rng: np.random.Generator,
    hotspot_fraction: float = 1.6,
) -> NDArray[np.float64]:
    """Create a fine truth whose block means exactly match the satellite field."""
    truth = coarse_to_fine(grid, satellite_field)
    for (_, _), block in grid.iter_coarse_blocks():
        local = truth[block].copy()
        mean_value = float(np.mean(local))
        local_variation = np.full(grid.fine_per_coarse, -hotspot_fraction * mean_value / (grid.cells_per_coarse - 1))
        hotspot_index = (int(rng.integers(0, grid.fine_per_coarse[0])), int(rng.integers(0, grid.fine_per_coarse[1])))
        local_variation[hotspot_index] = hotspot_fraction * mean_value
        truth[block] = np.maximum(local + local_variation, 0.0)
        truth[block] += mean_value - float(np.mean(truth[block]))
    return np.maximum(truth, 0.0)


def diffusion_laplacian(field: NDArray[np.float64]) -> NDArray[np.float64]:
    """Graph Laplacian on a 2-D grid with no-flux boundaries."""
    laplacian = np.zeros_like(field, dtype=np.float64)
    laplacian[:-1, :] += field[1:, :] - field[:-1, :]
    laplacian[1:, :] += field[:-1, :] - field[1:, :]
    laplacian[:, :-1] += field[:, 1:] - field[:, :-1]
    laplacian[:, 1:] += field[:, :-1] - field[:, 1:]
    return laplacian


def source_field(grid: GridSpec, step: int, strength: float) -> NDArray[np.float64]:
    """A small persistent source q_k that slowly drifts across the fine grid."""
    rows, cols = grid.fine_shape
    row_axis = np.arange(rows, dtype=np.float64)
    col_axis = np.arange(cols, dtype=np.float64)
    rr, cc = np.meshgrid(row_axis, col_axis, indexing="ij")
    center_row = rows * (0.62 + 0.08 * np.sin(0.15 * step))
    center_col = cols * (0.25 + 0.08 * np.cos(0.12 * step))
    return strength * np.exp(-((rr - center_row) ** 2 + (cc - center_col) ** 2) / 5.0)


def dynamics_step(
    field: NDArray[np.float64],
    control: NDArray[np.float64],
    source: NDArray[np.float64],
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Advance x_{k+1} = x_k + dt(DLx - lambda x + q - beta u) + w."""
    noise = 0.0
    if rng is not None and config.process_noise_std > 0.0:
        noise = rng.normal(0.0, config.process_noise_std, size=field.shape)
    next_field = field + config.dt * (
        config.diffusion_rate * diffusion_laplacian(field)
        - config.decay_rate * field
        + source
        - config.control_gain * control
    )
    return np.maximum(next_field + noise, 0.0)


def update_estimate_with_satellite(
    grid: GridSpec,
    estimate: NDArray[np.float64],
    satellite_observation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Uniformly distribute each coarse residual so H x_hat matches the satellite data."""
    residual = satellite_observation - fine_to_coarse(grid, estimate)
    return np.maximum(estimate + coarse_to_fine(grid, residual), 0.0)


def update_estimate_with_usv(
    estimate: NDArray[np.float64],
    observations: Sequence[tuple[tuple[int, int], float]],
    assimilation_gain: float,
) -> NDArray[np.float64]:
    """Overwrite observed fine cells with a gain-weighted local USV measurement."""
    updated = estimate.copy()
    for position, value in observations:
        updated[position] = (1.0 - assimilation_gain) * updated[position] + assimilation_gain * value
    return np.maximum(updated, 0.0)


def update_uncertainty(
    uncertainty: NDArray[np.float64],
    observations: Sequence[tuple[tuple[int, int], float]],
) -> NDArray[np.float64]:
    """Mark fine cells that have been directly observed by a USV."""
    updated = uncertainty.copy()
    for position, _ in observations:
        updated[position] = 0.0
    return updated


def update_observation_density(
    observation_density: NDArray[np.float64],
    observations: Sequence[tuple[tuple[int, int], float]],
    decay: float,
) -> NDArray[np.float64]:
    """Update z, the fine-grid observation density produced by USV visits."""
    updated = max(1.0 - decay, 0.0) * observation_density
    for position, _ in observations:
        updated[position] += 1.0
    return updated


def desired_output_from_satellite(satellite: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build y_ref from the coarse satellite importance distribution."""
    return normalize_distribution(satellite)


def output_from_observation_density(grid: GridSpec, observation_density: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build y = H z from fine-grid observation density."""
    return normalize_distribution(fine_to_coarse(grid, observation_density))


def normalize_distribution(field: NDArray[np.float64]) -> NDArray[np.float64]:
    total = float(np.sum(field))
    if total <= 0.0:
        return np.zeros_like(field, dtype=np.float64)
    return np.asarray(field, dtype=np.float64) / total


def smooth_estimate(
    grid: GridSpec,
    estimate: NDArray[np.float64],
    satellite_observation: NDArray[np.float64],
    smoothing_gain: float,
) -> NDArray[np.float64]:
    """Diffuse local USV corrections slightly while preserving satellite block means."""
    if smoothing_gain <= 0.0:
        return estimate
    smoothed = np.maximum(estimate + smoothing_gain * diffusion_laplacian(estimate), 0.0)
    return update_estimate_with_satellite(grid, smoothed, satellite_observation)


def ideal_control(estimate: NDArray[np.float64], config: SimulationConfig) -> NDArray[np.float64]:
    """Directly control every fine cell based on the estimated excess concentration."""
    excess = np.maximum(estimate - config.coarse_reference, 0.0)
    desired = excess / max(config.dt * config.control_gain, 1e-9)
    return np.minimum(desired, config.control_limit)


def usv_local_control(
    grid: GridSpec,
    estimate: NDArray[np.float64],
    usvs: Sequence[USV],
    config: SimulationConfig,
) -> NDArray[np.float64]:
    """Apply control only at each USV position and its four-neighbor cells."""
    control = np.zeros(grid.fine_shape, dtype=np.float64)
    for usv in usvs:
        reachable = grid.neighbors(usv.position, include_stay=True)
        values = np.array([estimate[position] for position in reachable], dtype=np.float64)
        excess = np.maximum(values - config.coarse_reference, 0.0)
        if float(np.sum(excess)) <= 0.0:
            continue
        weights = excess / float(np.sum(excess))
        for position, weight in zip(reachable, weights):
            control[position] += config.control_limit * weight
    return np.minimum(control, config.control_limit)


def move_usvs(
    grid: GridSpec,
    estimate: NDArray[np.float64],
    satellite: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    observation_density: NDArray[np.float64],
    usvs: Sequence[USV],
    config: SimulationConfig,
) -> list[USV]:
    """Move USVs with the configured one-step planner."""
    if config.planner_mode == "greedy":
        return _move_usvs_greedy(grid, estimate, usvs)
    if config.planner_mode == "tracking":
        return _move_usvs_tracking(grid, satellite, uncertainty, observation_density, usvs, config)
    return _move_usvs_satellite_survey(grid, satellite, uncertainty, usvs, config)


def _move_usvs_tracking(
    grid: GridSpec,
    satellite: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    observation_density: NDArray[np.float64],
    usvs: Sequence[USV],
    config: SimulationConfig,
) -> list[USV]:
    """Output-tracking planner for y = H z -> y_ref."""
    y_ref = desired_output_from_satellite(satellite)
    y_output = output_from_observation_density(grid, observation_density)
    next_usvs: list[USV] = []
    reserved: set[tuple[int, int]] = set()
    selected_targets: set[tuple[int, int]] = set()
    planned_density = observation_density.copy()

    for usv in usvs:
        target = _select_tracking_target(grid, y_ref, y_output, uncertainty, usv.position, selected_targets, config)
        selected_targets.add(target)
        next_position = _one_step_toward_best_unobserved_cell(grid, uncertainty, usv.position, target)
        if next_position in reserved:
            next_position = _fallback_unreserved_neighbor(grid, uncertainty, usv.position, reserved)
        planned_density[next_position] += 1.0
        y_output = output_from_observation_density(grid, planned_density)
        reserved.add(next_position)
        next_usvs.append(USV(position=next_position))

    return next_usvs


def _select_tracking_target(
    grid: GridSpec,
    y_ref: NDArray[np.float64],
    y_output: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    position: tuple[int, int],
    selected_targets: set[tuple[int, int]],
    config: SimulationConfig,
) -> tuple[int, int]:
    best_score = -np.inf
    best_target = grid.coarse_index_for_position(position)
    deficit = y_ref - y_output

    for coarse_index, block in grid.iter_coarse_blocks():
        center = grid.coarse_center(coarse_index)
        distance = abs(center[0] - position[0]) + abs(center[1] - position[1])
        coarse_uncertainty = float(np.mean(uncertainty[block]))
        already_selected_penalty = config.tracking_target_penalty if coarse_index in selected_targets else 0.0
        score = (
            float(deficit[coarse_index])
            + config.tracking_uncertainty_weight * coarse_uncertainty
            - config.tracking_distance_weight * distance
            - already_selected_penalty
        )
        if score > best_score:
            best_score = score
            best_target = coarse_index
    return best_target


def _move_usvs_greedy(grid: GridSpec, estimate: NDArray[np.float64], usvs: Sequence[USV]) -> list[USV]:
    """Greedy one-cell motion toward high estimated fine concentration."""
    next_usvs: list[USV] = []
    reserved: set[tuple[int, int]] = set()
    for usv in usvs:
        candidates = grid.neighbors(usv.position, include_stay=True)
        candidates = sorted(
            candidates,
            key=lambda position: (estimate[position], -abs(position[0]), -abs(position[1])),
        )
        for position in reversed(candidates):
            if position not in reserved:
                reserved.add(position)
                next_usvs.append(USV(position=position))
                break
    return next_usvs


def _move_usvs_satellite_survey(
    grid: GridSpec,
    satellite: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    usvs: Sequence[USV],
    config: SimulationConfig,
) -> list[USV]:
    """Survey planner: prioritize important coarse cells and unobserved fine cells."""
    next_usvs: list[USV] = []
    reserved: set[tuple[int, int]] = set()
    selected_targets: set[tuple[int, int]] = set()

    for usv in usvs:
        target = _select_survey_target(grid, satellite, uncertainty, usv.position, selected_targets, config)
        selected_targets.add(target)
        next_position = _one_step_toward_best_unobserved_cell(grid, uncertainty, usv.position, target)
        if next_position in reserved:
            next_position = _fallback_unreserved_neighbor(grid, uncertainty, usv.position, reserved)
        reserved.add(next_position)
        next_usvs.append(USV(position=next_position))
    return next_usvs


def _select_survey_target(
    grid: GridSpec,
    satellite: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    position: tuple[int, int],
    selected_targets: set[tuple[int, int]],
    config: SimulationConfig,
) -> tuple[int, int]:
    satellite_max = max(float(np.max(satellite)), 1e-9)
    has_unobserved_coarse_cell = any(
        float(np.mean(uncertainty[block])) > 0.0 for _, block in grid.iter_coarse_blocks()
    )
    best_score = -np.inf
    best_target = grid.coarse_index_for_position(position)
    for coarse_index, block in grid.iter_coarse_blocks():
        coarse_uncertainty = float(np.mean(uncertainty[block]))
        if has_unobserved_coarse_cell and coarse_uncertainty <= 0.0:
            continue
        coarse_importance = float(satellite[coarse_index]) / satellite_max
        center = grid.coarse_center(coarse_index)
        distance = abs(center[0] - position[0]) + abs(center[1] - position[1])
        already_selected_penalty = 0.4 if coarse_index in selected_targets else 0.0
        score = (
            coarse_importance * (0.25 + 0.75 * coarse_uncertainty)
            + config.survey_uncertainty_weight * coarse_uncertainty
            - config.survey_distance_weight * distance
            - already_selected_penalty
        )
        if score > best_score:
            best_score = score
            best_target = coarse_index
    return best_target


def _one_step_toward_best_unobserved_cell(
    grid: GridSpec,
    uncertainty: NDArray[np.float64],
    position: tuple[int, int],
    target_coarse: tuple[int, int],
) -> tuple[int, int]:
    _, block = next(item for item in grid.iter_coarse_blocks() if item[0] == target_coarse)
    target_cells = [
        (row, col)
        for row in range(block[0].start, block[0].stop)
        for col in range(block[1].start, block[1].stop)
        if uncertainty[row, col] > 0.0
    ]
    if not target_cells:
        target_cells = _all_unobserved_cells(uncertainty)
    if not target_cells:
        target_cells = [
            (row, col) for row in range(block[0].start, block[0].stop) for col in range(block[1].start, block[1].stop)
        ]
    target_cell = min(target_cells, key=lambda cell: abs(cell[0] - position[0]) + abs(cell[1] - position[1]))
    candidates = grid.neighbors(position, include_stay=target_cell == position)
    return min(candidates, key=lambda cell: abs(cell[0] - target_cell[0]) + abs(cell[1] - target_cell[1]))


def _fallback_unreserved_neighbor(
    grid: GridSpec,
    uncertainty: NDArray[np.float64],
    position: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> tuple[int, int]:
    candidates = [cell for cell in grid.neighbors(position, include_stay=True) if cell not in reserved]
    if not candidates:
        return position
    return max(candidates, key=lambda cell: uncertainty[cell])


def _all_unobserved_cells(uncertainty: NDArray[np.float64]) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(uncertainty.shape[0])
        for col in range(uncertainty.shape[1])
        if uncertainty[row, col] > 0.0
    ]


def observe_usvs(
    truth: NDArray[np.float64],
    usvs: Sequence[USV],
    rng: np.random.Generator,
    noise_std: float,
) -> list[tuple[tuple[int, int], float]]:
    observations: list[tuple[tuple[int, int], float]] = []
    for usv in usvs:
        noise = float(rng.normal(0.0, noise_std)) if noise_std > 0.0 else 0.0
        observations.append((usv.position, max(float(truth[usv.position] + noise), 0.0)))
    return observations


def run_simulation(config: SimulationConfig | None = None, grid: GridSpec | None = None) -> SimulationHistory:
    """Run the minimal coupled satellite-USV estimation and control simulation."""
    config = config or SimulationConfig()
    grid = grid or GridSpec()
    rng = np.random.default_rng(config.seed)

    initial_satellite = build_initial_satellite_field(grid)
    truth = build_satellite_consistent_truth(grid, initial_satellite, rng, config.hotspot_fraction)
    estimate = coarse_to_fine(grid, initial_satellite)
    uncertainty = np.ones(grid.fine_shape, dtype=np.float64)
    observation_density = np.zeros(grid.fine_shape, dtype=np.float64)
    usvs = _initial_usvs(grid, config.usv_count)
    latest_satellite = initial_satellite
    history = SimulationHistory()

    for step in range(config.steps + 1):
        satellite_noise = rng.normal(0.0, config.satellite_noise_std, size=grid.coarse_shape)
        if step % max(config.satellite_period, 1) == 0:
            latest_satellite = np.maximum(fine_to_coarse(grid, truth) + satellite_noise, 0.0)
            estimate = update_estimate_with_satellite(grid, estimate, latest_satellite)

        usv_observations = observe_usvs(truth, usvs, rng, config.usv_noise_std)
        estimate = update_estimate_with_usv(estimate, usv_observations, config.usv_assimilation_gain)
        uncertainty = update_uncertainty(uncertainty, usv_observations)
        observation_density = update_observation_density(
            observation_density,
            usv_observations,
            config.observation_density_decay,
        )
        estimate = smooth_estimate(grid, estimate, latest_satellite, config.estimator_smoothing_gain)

        control = _compute_control(grid, estimate, usvs, config)
        _record(history, grid, truth, estimate, latest_satellite, control, uncertainty, observation_density, usvs)

        if step == config.steps:
            break

        source = source_field(grid, step, config.source_strength)
        truth = dynamics_step(truth, control, source, config, rng)
        estimate = dynamics_step(estimate, control, source, config, None)
        usvs = move_usvs(grid, estimate, latest_satellite, uncertainty, observation_density, usvs, config)

    return history


def _compute_control(
    grid: GridSpec,
    estimate: NDArray[np.float64],
    usvs: Sequence[USV],
    config: SimulationConfig,
) -> NDArray[np.float64]:
    if config.control_mode == "none":
        return np.zeros(grid.fine_shape, dtype=np.float64)
    if config.control_mode == "ideal":
        return ideal_control(estimate, config)
    return usv_local_control(grid, estimate, usvs, config)


def _record(
    history: SimulationHistory,
    grid: GridSpec,
    truth: NDArray[np.float64],
    estimate: NDArray[np.float64],
    satellite: NDArray[np.float64],
    control: NDArray[np.float64],
    uncertainty: NDArray[np.float64],
    observation_density: NDArray[np.float64],
    usvs: Sequence[USV],
) -> None:
    y_ref = desired_output_from_satellite(satellite)
    y_output = output_from_observation_density(grid, observation_density)
    fine_weights = coarse_to_fine(grid, y_ref)
    fine_error = truth - estimate

    history.truth.append(truth.copy())
    history.estimate.append(estimate.copy())
    history.satellite.append(satellite.copy())
    history.control.append(control.copy())
    history.uncertainty.append(uncertainty.copy())
    history.observation_density.append(observation_density.copy())
    history.y_ref.append(y_ref.copy())
    history.y_output.append(y_output.copy())
    history.usv_positions.append([usv.position for usv in usvs])
    history.coarse_rmse.append(float(np.sqrt(np.mean((fine_to_coarse(grid, truth) - satellite) ** 2))))
    history.fine_rmse.append(float(np.sqrt(np.mean(fine_error**2))))
    weighted_mse = np.sum(fine_weights * fine_error**2) / max(np.sum(fine_weights), 1e-9)
    history.weighted_rmse.append(float(np.sqrt(weighted_mse)))
    history.tracking_error.append(float(np.sqrt(np.sum((y_output - y_ref) ** 2))))
    history.total_concentration.append(float(np.sum(truth)))
    history.observed_fraction.append(float(np.mean(uncertainty <= 0.0)))


def _initial_usvs(grid: GridSpec, usv_count: int) -> list[USV]:
    rows, cols = grid.fine_shape
    candidates = [
        (0, 0),
        (rows - 1, cols - 1),
        (rows - 1, 0),
        (0, cols - 1),
        (rows // 2, cols // 2),
    ]
    return [USV(position=grid.clip_position(candidates[i % len(candidates)])) for i in range(usv_count)]
