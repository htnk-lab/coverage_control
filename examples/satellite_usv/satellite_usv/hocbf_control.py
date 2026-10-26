#!/usr/bin/env python

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .cbf_control import build_cbf_desired_output, target_output_at_step
from .model import GridSpec, diffusion_laplacian, fine_to_coarse


@dataclass
class HOCBFConfig:
    """Configuration for the minimal HOCBF toy demo."""

    steps: int = 240
    dt: float = 0.05
    diffusion_rate: float = 0.015
    decay_rate: float = 0.0
    control_gain: float = 0.85
    control_limit: float = 1.2
    velocity_limit: float = 1.4
    w_limit: float = 2.5
    hocbf_epsilon: float = 0.08
    k1: float = 5.0
    k2: float = 5.0
    position_cbf_gain: float = 3.0
    position_barrier_power: int = 6
    action_cbf_gain: float = 5.0
    target_rate: float = 0.35
    velocity_nominal_gain: float = 2.8
    velocity_weight: float = 0.45
    action_rate_weight: float = 0.5
    action_decay_gain: float = 1.0
    slack_penalty: float = 1.0e7
    kernel_sigma: float = 0.85
    source_mode: str = "static"
    source_strength: float = 0.025
    source_sigma: float = 0.65
    source_period: float = 6.0
    source_target_weight: float = 8.0
    target_score_power: float = 2.0


@dataclass
class HOCBFHistory:
    """Recorded arrays from the HOCBF demo."""

    hocbf_epsilon: float = 0.0
    control_limit: float = 0.0
    velocity_limit: float = 0.0
    assigned_coarse: list[tuple[int, int]] = field(default_factory=list)
    truth: list[NDArray[np.float64]] = field(default_factory=list)
    coarse_output: list[NDArray[np.float64]] = field(default_factory=list)
    target_output: list[NDArray[np.float64]] = field(default_factory=list)
    desired_output: list[NDArray[np.float64]] = field(default_factory=list)
    source: list[NDArray[np.float64]] = field(default_factory=list)
    control: list[NDArray[np.float64]] = field(default_factory=list)
    positions: list[NDArray[np.float64]] = field(default_factory=list)
    actions: list[NDArray[np.float64]] = field(default_factory=list)
    velocities: list[NDArray[np.float64]] = field(default_factory=list)
    action_rates: list[NDArray[np.float64]] = field(default_factory=list)
    nominal_velocities: list[NDArray[np.float64]] = field(default_factory=list)
    nominal_targets: list[NDArray[np.float64]] = field(default_factory=list)
    barrier: list[NDArray[np.float64]] = field(default_factory=list)
    psi1: list[NDArray[np.float64]] = field(default_factory=list)
    hocbf_residual: list[NDArray[np.float64]] = field(default_factory=list)
    slack: list[NDArray[np.float64]] = field(default_factory=list)
    position_barrier: list[float] = field(default_factory=list)
    action_margin: list[float] = field(default_factory=list)
    max_upper_error: list[float] = field(default_factory=list)
    tracking_error: list[float] = field(default_factory=list)
    desired_error: list[float] = field(default_factory=list)
    total_concentration: list[float] = field(default_factory=list)
    cumulative_distance: list[NDArray[np.float64]] = field(default_factory=list)
    action_energy: list[float] = field(default_factory=list)
    velocity_energy: list[float] = field(default_factory=list)
    path_length: list[float] = field(default_factory=list)


def run_hocbf_simulation(
    config: HOCBFConfig | None = None,
    grid: GridSpec | None = None,
    initial_truth: NDArray[np.float64] | None = None,
) -> HOCBFHistory:
    """Run the minimal HOCBF simulation."""
    config = config or HOCBFConfig()
    grid = grid or GridSpec(coarse_shape=(2, 2), fine_per_coarse=(4, 4))
    if grid.coarse_shape != (2, 2) or grid.fine_per_coarse != (4, 4):
        raise ValueError(
            "The HOCBF demo is intentionally fixed to 2x2 coarse cells, "
            "with 4x4 fine cells inside each coarse cell."
        )

    truth = build_hocbf_truth(grid) if initial_truth is None else initial_truth.copy()
    initial_output = fine_to_coarse(grid, truth)
    desired_output = build_cbf_desired_output(initial_output)
    assigned_coarse = [coarse_index for coarse_index, _ in grid.iter_coarse_blocks()]
    positions = _initial_positions(grid, assigned_coarse)
    actions = np.zeros(len(assigned_coarse), dtype=np.float64)
    cumulative_distance = np.zeros(len(assigned_coarse), dtype=np.float64)
    action_energy = 0.0
    velocity_energy = 0.0
    path_length = 0.0
    history = HOCBFHistory(
        hocbf_epsilon=config.hocbf_epsilon,
        control_limit=config.control_limit,
        velocity_limit=config.velocity_limit,
        assigned_coarse=assigned_coarse,
    )

    for step in range(config.steps + 1):
        time = step * config.dt
        source, source_dot = source_field_and_dot(grid, time, config)
        target = target_output_at_step(initial_output, desired_output, step, config)
        target_dot = _target_dot(initial_output, desired_output, step, config)
        target_ddot = _target_ddot(initial_output, desired_output, step, config)
        solution = solve_hocbf_qp(
            grid,
            truth,
            positions,
            actions,
            assigned_coarse,
            target,
            target_dot,
            target_ddot,
            source,
            source_dot,
            config,
        )
        _record(
            history,
            grid,
            truth,
            target,
            desired_output,
            source,
            positions,
            actions,
            solution,
            cumulative_distance,
            action_energy,
            velocity_energy,
            path_length,
            config,
        )

        if step == config.steps:
            break

        natural_dot = _natural_dynamics(truth, source, config)
        truth = np.maximum(truth + config.dt * (natural_dot - config.control_gain * solution.control), 0.0)
        speed = np.linalg.norm(solution.velocities, axis=1)
        cumulative_distance = cumulative_distance + config.dt * speed
        action_energy += config.dt * float(actions @ actions)
        velocity_energy += config.dt * float(np.sum(solution.velocities**2))
        path_length += config.dt * float(np.sum(speed))
        positions = positions + config.dt * solution.velocities
        actions = actions + config.dt * solution.action_rates

    return history


@dataclass
class HOCBFSolution:
    """One-step HOCBF-QP solution."""

    velocities: NDArray[np.float64]
    action_rates: NDArray[np.float64]
    nominal_velocities: NDArray[np.float64]
    nominal_targets: NDArray[np.float64]
    control: NDArray[np.float64]
    barrier: NDArray[np.float64]
    psi1: NDArray[np.float64]
    hocbf_residual: NDArray[np.float64]
    slack: NDArray[np.float64]
    position_barrier: float
    action_margin: float


def solve_hocbf_qp(
    grid: GridSpec,
    truth: NDArray[np.float64],
    positions: NDArray[np.float64],
    actions: NDArray[np.float64],
    assigned_coarse: list[tuple[int, int]],
    target: NDArray[np.float64],
    target_dot: NDArray[np.float64],
    target_ddot: NDArray[np.float64],
    source: NDArray[np.float64],
    source_dot: NDArray[np.float64],
    config: HOCBFConfig,
) -> HOCBFSolution:
    """Solve the coupled HOCBF-QP for velocity and intervention-rate inputs."""
    robot_count = len(actions)
    coarse_count = grid.coarse_count
    kernels = kernel_matrix(grid, positions, config.kernel_sigma)
    natural_dot = _natural_dynamics(truth, source, config)
    control = np.einsum("ij,ijr,r->ij", truth, kernels, actions)
    x_dot = natural_dot - config.control_gain * control
    coarse_output = fine_to_coarse(grid, truth)
    af = fine_to_coarse(grid, natural_dot)
    s_matrix = _coarse_sensitivity(grid, truth, kernels)
    y_dot = af - config.control_gain * (s_matrix @ actions).reshape(grid.coarse_shape)
    b = target + config.hocbf_epsilon - coarse_output
    b_dot = target_dot - y_dot
    psi1 = b_dot + config.k1 * b

    cv = _velocity_coefficients(grid, truth, positions, actions, kernels, config)
    cw = config.control_gain * s_matrix
    constant = _hocbf_constant(grid, truth, x_dot, kernels, actions, target_ddot, source_dot, config)
    hocbf_base = constant + (config.k1 + config.k2) * b_dot.reshape(-1) + config.k1 * config.k2 * b.reshape(-1)
    nominal_targets = nominal_targets_from_efficiency(grid, truth, source, positions, assigned_coarse, target, config)
    nominal_velocities = config.velocity_nominal_gain * (nominal_targets - positions)
    nominal_velocities = np.clip(nominal_velocities, -config.velocity_limit, config.velocity_limit)
    nominal_action_rates = -config.action_decay_gain * actions

    velocities, action_rates, slack = _solve_qp(
        cv,
        cw,
        hocbf_base,
        positions,
        actions,
        assigned_coarse,
        nominal_velocities,
        nominal_action_rates,
        config,
        grid,
    )

    hocbf_residual = hocbf_base + cv @ velocities.reshape(-1) + cw @ action_rates
    return HOCBFSolution(
        velocities=velocities,
        action_rates=action_rates,
        nominal_velocities=nominal_velocities,
        nominal_targets=nominal_targets,
        control=control,
        barrier=b,
        psi1=psi1,
        hocbf_residual=hocbf_residual.reshape(grid.coarse_shape),
        slack=slack.reshape(grid.coarse_shape),
        position_barrier=_minimum_position_barrier(grid, positions, assigned_coarse, config),
        action_margin=float(min(np.min(actions), np.min(config.control_limit - actions))),
    )


def kernel_matrix(grid: GridSpec, positions: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Return K_jr(p_r) on the fine grid."""
    rows, cols = grid.fine_shape
    row_coords = np.arange(rows, dtype=np.float64)[:, None]
    col_coords = np.arange(cols, dtype=np.float64)[None, :]
    kernels = np.zeros((*grid.fine_shape, positions.shape[0]), dtype=np.float64)
    for index, position in enumerate(positions):
        distance_sq = (row_coords - position[0]) ** 2 + (col_coords - position[1]) ** 2
        kernels[:, :, index] = np.exp(-distance_sq / max(2.0 * sigma**2, 1e-9))
    return kernels


def build_hocbf_truth(grid: GridSpec) -> NDArray[np.float64]:
    """Create local hotspots near one corner of each coarse cell."""
    if grid.coarse_shape != (2, 2) or grid.fine_per_coarse != (4, 4):
        raise ValueError("build_hocbf_truth expects the 2x2 coarse / 8x8 fine grid used by the HOCBF demo.")
    rows, cols = grid.fine_shape
    row_axis = np.arange(rows, dtype=np.float64)
    col_axis = np.arange(cols, dtype=np.float64)
    rr, cc = np.meshgrid(row_axis, col_axis, indexing="ij")
    field = np.full(grid.fine_shape, 0.28, dtype=np.float64)
    for coarse_index, block in grid.iter_coarse_blocks():
        row_center, col_center = _local_hotspot_center(coarse_index, block)
        local_hotspot = np.exp(-((rr - row_center) ** 2 + (cc - col_center) ** 2) / 1.1)
        field += 1.15 * local_hotspot
    field += 0.06 * (rr / max(rows - 1, 1)) + 0.04 * (cc / max(cols - 1, 1))
    return np.asarray(field, dtype=np.float64)


def source_field_and_dot(
    grid: GridSpec,
    time: float,
    config: HOCBFConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the local environmental source and its time derivative."""
    source = np.zeros(grid.fine_shape, dtype=np.float64)
    source_dot = np.zeros(grid.fine_shape, dtype=np.float64)
    if config.source_strength <= 0.0 or config.source_mode == "none":
        return source, source_dot
    if config.source_mode not in {"static", "moving"}:
        raise ValueError("source_mode must be one of: 'static', 'moving', 'none'")

    rows = np.arange(grid.fine_shape[0], dtype=np.float64)[:, None]
    cols = np.arange(grid.fine_shape[1], dtype=np.float64)[None, :]
    sigma_sq = max(config.source_sigma**2, 1e-9)
    for robot_index, (coarse_index, _) in enumerate(grid.iter_coarse_blocks()):
        if config.source_mode == "moving":
            center, center_dot = _moving_source_center(grid, coarse_index, robot_index, time, config)
        else:
            center = _static_source_center(grid, coarse_index)
            center_dot = np.zeros(2, dtype=np.float64)
        row_delta = rows - center[0]
        col_delta = cols - center[1]
        gaussian = np.exp(-(row_delta**2 + col_delta**2) / (2.0 * sigma_sq))
        source += config.source_strength * gaussian
        source_dot += config.source_strength * gaussian * (
            row_delta * center_dot[0] + col_delta * center_dot[1]
        ) / sigma_sq
    return source, source_dot


def moving_source_field_and_dot(
    grid: GridSpec,
    time: float,
    config: HOCBFConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compatibility wrapper for the optional moving-source stress test."""
    moving_config = HOCBFConfig(**{**config.__dict__, "source_mode": "moving"})
    return source_field_and_dot(grid, time, moving_config)


def nominal_targets_from_efficiency(
    grid: GridSpec,
    truth: NDArray[np.float64],
    source: NDArray[np.float64],
    positions: NDArray[np.float64],
    assigned_coarse: list[tuple[int, int]],
    target: NDArray[np.float64],
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    """Choose nominal targets by maximizing same-coarse intervention efficiency."""
    coarse_output = fine_to_coarse(grid, truth)
    coarse_need = np.maximum(coarse_output - target, 0.0)
    nominal_targets = np.zeros_like(positions, dtype=np.float64)
    for robot_index, coarse_index in enumerate(assigned_coarse):
        row_slice, col_slice = _block_for_coarse(grid, coarse_index)
        local = truth[row_slice, col_slice]
        local_source = source[row_slice, col_slice]
        rows = np.arange(row_slice.start, row_slice.stop, dtype=np.float64)
        cols = np.arange(col_slice.start, col_slice.stop, dtype=np.float64)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        weight = 0.25 + float(coarse_need[coarse_index])
        scores = weight * (local + config.source_target_weight * local_source)
        smooth_weights = np.maximum(scores, 0.0) ** config.target_score_power
        total_weight = max(float(np.sum(smooth_weights)), 1e-9)
        target_row = float(np.sum(smooth_weights * rr) / total_weight)
        target_col = float(np.sum(smooth_weights * cc) / total_weight)
        row_min, row_max, col_min, col_max = _continuous_bounds_for_coarse(grid, coarse_index)
        nominal_targets[robot_index] = [
            float(np.clip(target_row, row_min + 0.35, row_max - 0.35)),
            float(np.clip(target_col, col_min + 0.35, col_max - 0.35)),
        ]
    return nominal_targets


def efficiency_gradient_velocity(
    grid: GridSpec,
    truth: NDArray[np.float64],
    positions: NDArray[np.float64],
    assigned_coarse: list[tuple[int, int]],
    target: NDArray[np.float64],
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    """Gradient of G_r = sum_i omega_i S_ir, kept for experiments and reporting."""
    coarse_output = fine_to_coarse(grid, truth)
    weights = np.maximum(coarse_output - target, 0.0).reshape(-1)
    rows = np.arange(grid.fine_shape[0], dtype=np.float64)[:, None]
    cols = np.arange(grid.fine_shape[1], dtype=np.float64)[None, :]
    velocity = np.zeros_like(positions, dtype=np.float64)
    for robot_index, position in enumerate(positions):
        kernel = kernel_matrix(grid, positions[[robot_index]], config.kernel_sigma)[:, :, 0]
        grad_row = kernel * (rows - position[0]) / config.kernel_sigma**2
        grad_col = kernel * (cols - position[1]) / config.kernel_sigma**2
        grad_by_coarse = np.zeros((grid.coarse_count, 2), dtype=np.float64)
        for coarse_flat, (_, block) in enumerate(grid.iter_coarse_blocks()):
            grad_by_coarse[coarse_flat, 0] = float(np.mean(truth[block] * grad_row[block]))
            grad_by_coarse[coarse_flat, 1] = float(np.mean(truth[block] * grad_col[block]))
        gradient = weights @ grad_by_coarse
        if not np.any(weights > 0.0):
            coarse_flat = assigned_coarse.index(assigned_coarse[robot_index])
            gradient = grad_by_coarse[coarse_flat]
        velocity[robot_index] = config.velocity_nominal_gain * gradient
    return np.clip(velocity, -config.velocity_limit, config.velocity_limit)


def _solve_qp(
    cv: NDArray[np.float64],
    cw: NDArray[np.float64],
    hocbf_base: NDArray[np.float64],
    positions: NDArray[np.float64],
    actions: NDArray[np.float64],
    assigned_coarse: list[tuple[int, int]],
    nominal_velocities: NDArray[np.float64],
    nominal_action_rates: NDArray[np.float64],
    config: HOCBFConfig,
    grid: GridSpec,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    robot_count = actions.size
    coarse_count = hocbf_base.size
    variable_count = 2 * robot_count + robot_count + coarse_count
    initial = np.zeros(variable_count, dtype=np.float64)
    initial[: 2 * robot_count] = nominal_velocities.reshape(-1)
    bounds = [(-config.velocity_limit, config.velocity_limit)] * (2 * robot_count)
    bounds += [(-config.w_limit, config.w_limit)] * robot_count
    bounds += [(0.0, None)] * coarse_count

    def unpack(variables: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        velocity = variables[: 2 * robot_count].reshape(robot_count, 2)
        action_rate = variables[2 * robot_count : 3 * robot_count]
        slack = variables[3 * robot_count :]
        return velocity, action_rate, slack

    def objective(variables: NDArray[np.float64]) -> float:
        velocity, action_rate, slack = unpack(variables)
        velocity_error = (velocity - nominal_velocities).reshape(-1)
        action_rate_error = action_rate - nominal_action_rates
        return (
            0.5 * config.velocity_weight * float(velocity_error @ velocity_error)
            + 0.5 * config.action_rate_weight * float(action_rate_error @ action_rate_error)
            + 0.5 * config.slack_penalty * float(slack @ slack)
        )

    def jacobian(variables: NDArray[np.float64]) -> NDArray[np.float64]:
        velocity, action_rate, slack = unpack(variables)
        gradient = np.zeros_like(variables)
        gradient[: 2 * robot_count] = config.velocity_weight * (velocity - nominal_velocities).reshape(-1)
        gradient[2 * robot_count : 3 * robot_count] = config.action_rate_weight * (action_rate - nominal_action_rates)
        gradient[3 * robot_count :] = config.slack_penalty * slack
        return gradient

    def constraints(variables: NDArray[np.float64]) -> NDArray[np.float64]:
        velocity, action_rate, slack = unpack(variables)
        hocbf = hocbf_base + cv @ velocity.reshape(-1) + cw @ action_rate + slack
        return np.concatenate(
            [
                hocbf,
                _position_cbf_constraints(grid, positions, velocity, assigned_coarse, config),
                _action_cbf_constraints(actions, action_rate, config),
            ]
        )

    result = minimize(
        objective,
        initial,
        jac=jacobian,
        bounds=bounds,
        constraints={"type": "ineq", "fun": constraints},
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 200, "disp": False},
    )
    variables = np.asarray(result.x if result.success else initial, dtype=np.float64)
    velocities, action_rates, slack = unpack(variables)
    return velocities, action_rates, np.maximum(slack, 0.0)


def _coarse_sensitivity(
    grid: GridSpec,
    truth: NDArray[np.float64],
    kernels: NDArray[np.float64],
) -> NDArray[np.float64]:
    sensitivity = np.zeros((grid.coarse_count, kernels.shape[-1]), dtype=np.float64)
    for coarse_flat, (_, block) in enumerate(grid.iter_coarse_blocks()):
        weighted = truth[block][:, :, None] * kernels[block]
        sensitivity[coarse_flat, :] = np.mean(weighted, axis=(0, 1))
    return sensitivity


def _velocity_coefficients(
    grid: GridSpec,
    truth: NDArray[np.float64],
    positions: NDArray[np.float64],
    actions: NDArray[np.float64],
    kernels: NDArray[np.float64],
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    coeffs = np.zeros((grid.coarse_count, 2 * actions.size), dtype=np.float64)
    row_coords = np.arange(grid.fine_shape[0], dtype=np.float64)[:, None]
    col_coords = np.arange(grid.fine_shape[1], dtype=np.float64)[None, :]
    for robot_index, position in enumerate(positions):
        row_factor = kernels[:, :, robot_index] * (row_coords - position[0]) / config.kernel_sigma**2
        col_factor = kernels[:, :, robot_index] * (col_coords - position[1]) / config.kernel_sigma**2
        for coarse_flat, (_, block) in enumerate(grid.iter_coarse_blocks()):
            coeffs[coarse_flat, 2 * robot_index] = config.control_gain * actions[robot_index] * float(
                np.mean(truth[block] * row_factor[block])
            )
            coeffs[coarse_flat, 2 * robot_index + 1] = config.control_gain * actions[robot_index] * float(
                np.mean(truth[block] * col_factor[block])
            )
    return coeffs


def _hocbf_constant(
    grid: GridSpec,
    truth: NDArray[np.float64],
    x_dot: NDArray[np.float64],
    kernels: NDArray[np.float64],
    actions: NDArray[np.float64],
    target_ddot: NDArray[np.float64],
    source_dot: NDArray[np.float64],
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    jf_xdot = config.diffusion_rate * diffusion_laplacian(x_dot) - config.decay_rate * x_dot + source_dot
    sdot_from_x = np.einsum("ij,ijr,r->ij", x_dot, kernels, actions)
    constant = target_ddot - fine_to_coarse(grid, jf_xdot)
    constant += config.control_gain * fine_to_coarse(grid, sdot_from_x)
    return constant.reshape(-1)


def _position_cbf_constraints(
    grid: GridSpec,
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    assigned_coarse: list[tuple[int, int]],
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    constraints: list[float] = []
    for position, velocity, coarse_index in zip(positions, velocities, assigned_coarse):
        barrier, gradient = _position_barrier_and_gradient(grid, position, coarse_index, config)
        constraints.append(float(gradient @ velocity + config.position_cbf_gain * barrier))
    return np.asarray(constraints, dtype=np.float64)


def _action_cbf_constraints(
    actions: NDArray[np.float64],
    action_rates: NDArray[np.float64],
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    return np.concatenate(
        [
            action_rates + config.action_cbf_gain * actions,
            -action_rates + config.action_cbf_gain * (config.control_limit - actions),
        ]
    )


def _target_dot(
    initial_output: NDArray[np.float64],
    desired_output: NDArray[np.float64],
    step: int,
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    time = step * config.dt
    return -config.target_rate * (initial_output - desired_output) * np.exp(-config.target_rate * time)


def _target_ddot(
    initial_output: NDArray[np.float64],
    desired_output: NDArray[np.float64],
    step: int,
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    time = step * config.dt
    return config.target_rate**2 * (initial_output - desired_output) * np.exp(-config.target_rate * time)


def _natural_dynamics(
    field: NDArray[np.float64],
    source: NDArray[np.float64],
    config: HOCBFConfig,
) -> NDArray[np.float64]:
    return config.diffusion_rate * diffusion_laplacian(field) - config.decay_rate * field + source


def _initial_positions(grid: GridSpec, assigned_coarse: list[tuple[int, int]]) -> NDArray[np.float64]:
    positions = []
    for coarse_index in assigned_coarse:
        row_min, row_max, col_min, col_max = _continuous_bounds_for_coarse(grid, coarse_index)
        positions.append([row_min + 0.35 * (row_max - row_min), col_min + 0.25 * (col_max - col_min)])
    return np.asarray(positions, dtype=np.float64)


def _block_for_coarse(grid: GridSpec, coarse_index: tuple[int, int]) -> tuple[slice, slice]:
    return next(block for index, block in grid.iter_coarse_blocks() if index == coarse_index)


def _continuous_bounds_for_coarse(
    grid: GridSpec,
    coarse_index: tuple[int, int],
) -> tuple[float, float, float, float]:
    row_slice, col_slice = _block_for_coarse(grid, coarse_index)
    return (
        float(row_slice.start),
        float(row_slice.stop - 1),
        float(col_slice.start),
        float(col_slice.stop - 1),
    )


def _moving_source_center(
    grid: GridSpec,
    coarse_index: tuple[int, int],
    robot_index: int,
    time: float,
    config: HOCBFConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    row_min, row_max, col_min, col_max = _continuous_bounds_for_coarse(grid, coarse_index)
    center = np.asarray([(row_min + row_max) / 2.0, (col_min + col_max) / 2.0], dtype=np.float64)
    half_width = np.asarray([(row_max - row_min) / 2.0, (col_max - col_min) / 2.0], dtype=np.float64)
    radius = 0.72 * half_width
    omega = 2.0 * np.pi / max(config.source_period, 1e-9)
    phase = omega * time + 0.75 * robot_index
    offset = radius * np.asarray([np.cos(phase), np.sin(phase + 0.4 * robot_index)], dtype=np.float64)
    offset_dot = radius * omega * np.asarray(
        [-np.sin(phase), np.cos(phase + 0.4 * robot_index)],
        dtype=np.float64,
    )
    return center + offset, offset_dot


def _static_source_center(grid: GridSpec, coarse_index: tuple[int, int]) -> NDArray[np.float64]:
    block = _block_for_coarse(grid, coarse_index)
    return np.asarray(_local_hotspot_center(coarse_index, block), dtype=np.float64)


def _local_hotspot_center(coarse_index: tuple[int, int], block: tuple[slice, slice]) -> tuple[float, float]:
    row_center = block[0].stop - 0.85
    col_center = block[1].stop - 0.85
    if coarse_index == (0, 1):
        row_center = block[0].stop - 0.75
        col_center = block[1].start + 0.75
    if coarse_index == (1, 0):
        row_center = block[0].start + 0.75
        col_center = block[1].stop - 0.75
    return float(row_center), float(col_center)


def _position_barrier_and_gradient(
    grid: GridSpec,
    position: NDArray[np.float64],
    coarse_index: tuple[int, int],
    config: HOCBFConfig,
) -> tuple[float, NDArray[np.float64]]:
    row_min, row_max, col_min, col_max = _continuous_bounds_for_coarse(grid, coarse_index)
    center = np.asarray([(row_min + row_max) / 2.0, (col_min + col_max) / 2.0], dtype=np.float64)
    radii = np.asarray([(row_max - row_min) / 2.0, (col_max - col_min) / 2.0], dtype=np.float64)
    radii = np.maximum(radii, 1e-9)
    shifted = position - center
    normalized = shifted / radii
    power = config.position_barrier_power
    barrier = 1.0 - float(np.sum(np.abs(normalized) ** power))
    gradient = -power * np.sign(normalized) * np.abs(normalized) ** (power - 1) / radii
    return barrier, gradient


def _minimum_position_barrier(
    grid: GridSpec,
    positions: NDArray[np.float64],
    assigned_coarse: list[tuple[int, int]],
    config: HOCBFConfig,
) -> float:
    margins: list[float] = []
    for position, coarse_index in zip(positions, assigned_coarse):
        barrier, _ = _position_barrier_and_gradient(grid, position, coarse_index, config)
        margins.append(barrier)
    return float(min(margins))


def _record(
    history: HOCBFHistory,
    grid: GridSpec,
    truth: NDArray[np.float64],
    target: NDArray[np.float64],
    desired_output: NDArray[np.float64],
    source: NDArray[np.float64],
    positions: NDArray[np.float64],
    actions: NDArray[np.float64],
    solution: HOCBFSolution,
    cumulative_distance: NDArray[np.float64],
    action_energy: float,
    velocity_energy: float,
    path_length: float,
    config: HOCBFConfig,
) -> None:
    coarse_output = fine_to_coarse(grid, truth)
    upper_error = coarse_output - target
    history.truth.append(truth.copy())
    history.coarse_output.append(coarse_output.copy())
    history.target_output.append(target.copy())
    history.desired_output.append(desired_output.copy())
    history.source.append(source.copy())
    history.control.append(solution.control.copy())
    history.positions.append(positions.copy())
    history.actions.append(actions.copy())
    history.velocities.append(solution.velocities.copy())
    history.action_rates.append(solution.action_rates.copy())
    history.nominal_velocities.append(solution.nominal_velocities.copy())
    history.nominal_targets.append(solution.nominal_targets.copy())
    history.barrier.append(solution.barrier.copy())
    history.psi1.append(solution.psi1.copy())
    history.hocbf_residual.append(solution.hocbf_residual.copy())
    history.slack.append(solution.slack.copy())
    history.position_barrier.append(solution.position_barrier)
    history.action_margin.append(solution.action_margin)
    history.max_upper_error.append(float(np.max(upper_error)))
    history.tracking_error.append(float(np.sqrt(np.sum(upper_error**2))))
    history.desired_error.append(float(np.sqrt(np.sum((coarse_output - desired_output) ** 2))))
    history.total_concentration.append(float(np.sum(truth)))
    history.cumulative_distance.append(cumulative_distance.copy())
    history.action_energy.append(action_energy)
    history.velocity_energy.append(velocity_energy)
    history.path_length.append(path_length)
