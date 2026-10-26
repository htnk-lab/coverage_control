#!/usr/bin/env python

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .model import GridSpec, USV, diffusion_laplacian, fine_to_coarse


@dataclass
class CBFConfig:
    """Configuration for the first-order CBF toy demo."""

    steps: int = 80
    dt: float = 0.15
    diffusion_rate: float = 0.015
    decay_rate: float = 0.0
    control_gain: float = 0.85
    control_limit: float = 2.0
    cbf_gamma: float = 6.0
    cbf_epsilon: float = 0.08
    target_rate: float = 0.55
    slack_penalty: float = 1.0e6
    kernel_sigma: float = 1.6


@dataclass
class CBFHistory:
    """Recorded arrays from the first-order CBF toy demo."""

    cbf_epsilon: float = 0.0
    control_limit: float = 0.0
    truth: list[NDArray[np.float64]] = field(default_factory=list)
    coarse_output: list[NDArray[np.float64]] = field(default_factory=list)
    target_output: list[NDArray[np.float64]] = field(default_factory=list)
    desired_output: list[NDArray[np.float64]] = field(default_factory=list)
    control: list[NDArray[np.float64]] = field(default_factory=list)
    cbf_input: list[NDArray[np.float64]] = field(default_factory=list)
    barrier: list[NDArray[np.float64]] = field(default_factory=list)
    cbf_margin: list[NDArray[np.float64]] = field(default_factory=list)
    slack: list[NDArray[np.float64]] = field(default_factory=list)
    discrete_cbf_residual: list[NDArray[np.float64]] = field(default_factory=list)
    cbf_violation: list[float] = field(default_factory=list)
    usv_positions: list[list[tuple[int, int]]] = field(default_factory=list)
    assigned_coarse: list[tuple[int, int]] = field(default_factory=list)
    tracking_error: list[float] = field(default_factory=list)
    max_abs_error: list[float] = field(default_factory=list)
    desired_error: list[float] = field(default_factory=list)
    total_concentration: list[float] = field(default_factory=list)
    max_concentration: list[float] = field(default_factory=list)


def run_cbf_simulation(
    config: CBFConfig | None = None,
    grid: GridSpec | None = None,
    initial_truth: NDArray[np.float64] | None = None,
) -> CBFHistory:
    """Run a 2x2 coarse / 8x8 fine first-order CBF simulation."""
    config = config or CBFConfig()
    grid = grid or GridSpec(coarse_shape=(2, 2), fine_per_coarse=(4, 4))
    if grid.coarse_shape != (2, 2) or grid.fine_per_coarse != (4, 4):
        raise ValueError(
            "The first-order CBF demo is intentionally fixed to 2x2 coarse cells, "
            "with 4x4 fine cells inside each coarse cell."
        )

    truth = build_cbf_truth(grid) if initial_truth is None else initial_truth.copy()
    initial_output = fine_to_coarse(grid, truth)
    desired_output = build_cbf_desired_output(initial_output)
    assigned_coarse = [coarse_index for coarse_index, _ in grid.iter_coarse_blocks()]
    usvs = _initial_cbf_usvs(grid, assigned_coarse)
    history = CBFHistory(
        cbf_epsilon=config.cbf_epsilon,
        control_limit=config.control_limit,
        assigned_coarse=assigned_coarse,
    )

    for step in range(config.steps + 1):
        target_output = target_output_at_step(initial_output, desired_output, step, config)
        target_dot = target_output_dot_at_step(initial_output, desired_output, step, config)
        natural_dot = _natural_dynamics(truth, config)
        natural_coarse_dot = fine_to_coarse(grid, natural_dot)
        cbf_input, control, barrier, margin, slack = solve_first_order_cbf_control(
            grid,
            truth,
            natural_coarse_dot,
            target_output,
            target_dot,
            usvs,
            config,
        )
        next_truth = np.maximum(truth + config.dt * (natural_dot - config.control_gain * control), 0.0)
        next_target_output = target_output_at_step(initial_output, desired_output, step + 1, config)
        next_barrier = _barrier(fine_to_coarse(grid, next_truth), next_target_output, config)
        discrete_residual = next_barrier - (1.0 - config.cbf_gamma * config.dt) * barrier
        _record(
            history,
            grid,
            truth,
            target_output,
            desired_output,
            control,
            cbf_input,
            barrier,
            margin,
            slack,
            discrete_residual,
            usvs,
        )

        if step == config.steps:
            break

        truth = next_truth
        usvs = move_usvs_inside_assigned_coarse(grid, truth, usvs, assigned_coarse, step)

    return history


def build_cbf_truth(grid: GridSpec) -> NDArray[np.float64]:
    """Create a tiny fine field with nonuniform values inside each coarse cell."""
    if grid.coarse_shape != (2, 2) or grid.fine_per_coarse != (4, 4):
        raise ValueError("build_cbf_truth expects the 2x2 coarse / 8x8 fine grid used by the CBF demo.")
    rows, cols = grid.fine_shape
    row_axis = np.linspace(-1.0, 1.0, rows)
    col_axis = np.linspace(-1.0, 1.0, cols)
    rr, cc = np.meshgrid(row_axis, col_axis, indexing="ij")
    field = 0.62
    field += 0.22 * (cc + 1.0)
    field += 0.12 * (rr + 1.0)
    field += 0.24 * np.exp(-((rr + 0.35) ** 2 + (cc - 0.45) ** 2) / 0.18)
    field += 0.16 * np.exp(-((rr - 0.55) ** 2 + (cc + 0.55) ** 2) / 0.22)
    field += 0.05 * np.sin(3.0 * np.pi * (rr + 1.0)) * np.cos(2.0 * np.pi * (cc + 1.0))
    return np.maximum(np.asarray(field, dtype=np.float64), 0.0)


def build_cbf_desired_output(initial_output: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lower each coarse target by a different amount while keeping it feasible."""
    reduction_fraction = np.array([[0.34, 0.24], [0.48, 0.38]], dtype=np.float64)
    return np.maximum(initial_output * (1.0 - reduction_fraction), 0.0)


def target_output_at_step(
    initial_output: NDArray[np.float64],
    desired_output: NDArray[np.float64],
    step: int,
    config: CBFConfig,
) -> NDArray[np.float64]:
    """Smooth reference generated by the upper planner."""
    time = step * config.dt
    return desired_output + (initial_output - desired_output) * np.exp(-config.target_rate * time)


def target_output_dot_at_step(
    initial_output: NDArray[np.float64],
    desired_output: NDArray[np.float64],
    step: int,
    config: CBFConfig,
) -> NDArray[np.float64]:
    """Time derivative of the smooth reference."""
    time = step * config.dt
    return -config.target_rate * (initial_output - desired_output) * np.exp(-config.target_rate * time)


def solve_first_order_cbf_control(
    grid: GridSpec,
    truth: NDArray[np.float64],
    natural_coarse_dot: NDArray[np.float64],
    target_output: NDArray[np.float64],
    target_dot: NDArray[np.float64],
    usvs: list[USV],
    config: CBFConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solve the coupled first-order CBF-QP over all robot intervention amounts."""
    coarse_output = fine_to_coarse(grid, truth)
    error = coarse_output - target_output
    barrier = _barrier(coarse_output, target_output, config)
    kernels = np.stack([local_kernel(grid, usv.position, config.kernel_sigma) for usv in usvs], axis=-1)
    sensitivity = _coarse_sensitivity(grid, truth, kernels)

    base = -2.0 * error.reshape(-1) * (natural_coarse_dot.reshape(-1) - target_dot.reshape(-1))
    coef = 2.0 * config.control_gain * error.reshape(-1)[:, None] * sensitivity
    barrier_flat = barrier.reshape(-1)

    cbf_input_flat, slack_flat = _solve_minimum_intervention_qp(base, coef, barrier_flat, config)
    removal = np.einsum("ijr,r->ij", truth[:, :, None] * kernels, cbf_input_flat)
    margin_flat = base + coef @ cbf_input_flat + config.cbf_gamma * barrier_flat

    cbf_input = np.zeros(grid.coarse_shape, dtype=np.float64)
    for amount, usv in zip(cbf_input_flat, usvs):
        cbf_input[grid.coarse_index_for_position(usv.position)] += amount

    return (
        cbf_input,
        removal,
        barrier,
        margin_flat.reshape(grid.coarse_shape),
        slack_flat.reshape(grid.coarse_shape),
    )


def local_kernel(
    grid: GridSpec,
    position: tuple[int, int],
    sigma: float,
) -> NDArray[np.float64]:
    """Gaussian-like local actuation support over the whole fine grid."""
    kernel = np.zeros(grid.fine_shape, dtype=np.float64)
    for row in range(grid.fine_shape[0]):
        for col in range(grid.fine_shape[1]):
            distance_sq = float((row - position[0]) ** 2 + (col - position[1]) ** 2)
            kernel[row, col] = np.exp(-distance_sq / max(2.0 * sigma**2, 1e-9))
    return kernel


def move_usvs_inside_assigned_coarse(
    grid: GridSpec,
    truth: NDArray[np.float64],
    usvs: list[USV],
    assigned_coarse: list[tuple[int, int]],
    step: int,
) -> list[USV]:
    """Move each robot only inside its assigned coarse cell; this is not a CBF constraint."""
    next_usvs: list[USV] = []
    for robot_index, (usv, coarse_index) in enumerate(zip(usvs, assigned_coarse)):
        cells = _cells_in_coarse(grid, coarse_index)
        candidates = [
            cell
            for cell in grid.neighbors(usv.position, include_stay=False)
            if grid.coarse_index_for_position(cell) == coarse_index
        ]
        if not candidates:
            candidates = [usv.position]
        next_position = max(
            candidates,
            key=lambda cell: (
                truth[cell],
                -abs(cell[0] - usv.position[0]) - abs(cell[1] - usv.position[1]),
                -((cells.index(cell) - step - robot_index) % len(cells)),
            ),
        )
        next_usvs.append(USV(position=next_position))
    return next_usvs


def _solve_minimum_intervention_qp(
    base: NDArray[np.float64],
    coef: NDArray[np.float64],
    barrier: NDArray[np.float64],
    config: CBFConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    robot_count = coef.shape[1]
    coarse_count = coef.shape[0]
    initial = np.zeros(robot_count + coarse_count, dtype=np.float64)
    bounds = [(0.0, config.control_limit)] * robot_count + [(0.0, None)] * coarse_count

    def objective(variables: NDArray[np.float64]) -> float:
        actions = variables[:robot_count]
        slack = variables[robot_count:]
        return 0.5 * float(actions @ actions) + 0.5 * config.slack_penalty * float(slack @ slack)

    def jacobian(variables: NDArray[np.float64]) -> NDArray[np.float64]:
        gradient = np.zeros_like(variables)
        gradient[:robot_count] = variables[:robot_count]
        gradient[robot_count:] = config.slack_penalty * variables[robot_count:]
        return gradient

    def cbf_constraint(variables: NDArray[np.float64]) -> NDArray[np.float64]:
        actions = variables[:robot_count]
        slack = variables[robot_count:]
        return base + coef @ actions + config.cbf_gamma * barrier + slack

    result = minimize(
        objective,
        initial,
        jac=jacobian,
        bounds=bounds,
        constraints={"type": "ineq", "fun": cbf_constraint},
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 100, "disp": False},
    )
    variables = np.asarray(result.x if result.success else initial, dtype=np.float64)
    actions = np.clip(variables[:robot_count], 0.0, config.control_limit)
    slack = np.maximum(variables[robot_count:], 0.0)
    return actions, slack


def _coarse_sensitivity(
    grid: GridSpec,
    truth: NDArray[np.float64],
    kernels: NDArray[np.float64],
) -> NDArray[np.float64]:
    sensitivity = np.zeros((grid.coarse_count, kernels.shape[-1]), dtype=np.float64)
    for coarse_flat, (_, block) in enumerate(grid.iter_coarse_blocks()):
        weighted_kernel = truth[block][:, :, None] * kernels[block]
        sensitivity[coarse_flat, :] = np.mean(weighted_kernel, axis=(0, 1))
    return sensitivity


def _barrier(
    coarse_output: NDArray[np.float64],
    target_output: NDArray[np.float64],
    config: CBFConfig,
) -> NDArray[np.float64]:
    return config.cbf_epsilon**2 - (coarse_output - target_output) ** 2


def _natural_dynamics(field: NDArray[np.float64], config: CBFConfig) -> NDArray[np.float64]:
    return config.diffusion_rate * diffusion_laplacian(field) - config.decay_rate * field


def _initial_cbf_usvs(grid: GridSpec, assigned_coarse: list[tuple[int, int]]) -> list[USV]:
    return [USV(position=_cells_in_coarse(grid, coarse_index)[0]) for coarse_index in assigned_coarse]


def _block_for_coarse(grid: GridSpec, coarse_index: tuple[int, int]) -> tuple[slice, slice]:
    return next(block for index, block in grid.iter_coarse_blocks() if index == coarse_index)


def _cells_in_coarse(grid: GridSpec, coarse_index: tuple[int, int]) -> list[tuple[int, int]]:
    row_slice, col_slice = _block_for_coarse(grid, coarse_index)
    return [
        (row, col)
        for row in range(row_slice.start, row_slice.stop)
        for col in range(col_slice.start, col_slice.stop)
    ]


def _record(
    history: CBFHistory,
    grid: GridSpec,
    truth: NDArray[np.float64],
    target_output: NDArray[np.float64],
    desired_output: NDArray[np.float64],
    control: NDArray[np.float64],
    cbf_input: NDArray[np.float64],
    barrier: NDArray[np.float64],
    margin: NDArray[np.float64],
    slack: NDArray[np.float64],
    discrete_residual: NDArray[np.float64],
    usvs: list[USV],
) -> None:
    coarse_output = fine_to_coarse(grid, truth)
    error = coarse_output - target_output
    history.truth.append(truth.copy())
    history.coarse_output.append(coarse_output.copy())
    history.target_output.append(target_output.copy())
    history.desired_output.append(desired_output.copy())
    history.control.append(control.copy())
    history.cbf_input.append(cbf_input.copy())
    history.barrier.append(barrier.copy())
    history.cbf_margin.append(margin.copy())
    history.slack.append(slack.copy())
    history.discrete_cbf_residual.append(discrete_residual.copy())
    history.cbf_violation.append(float(np.max(np.maximum(-margin, 0.0))))
    history.usv_positions.append([usv.position for usv in usvs])
    history.tracking_error.append(float(np.sqrt(np.sum(error**2))))
    history.max_abs_error.append(float(np.max(np.abs(error))))
    history.desired_error.append(float(np.sqrt(np.sum((coarse_output - desired_output) ** 2))))
    history.total_concentration.append(float(np.sum(truth)))
    history.max_concentration.append(float(np.max(truth)))
