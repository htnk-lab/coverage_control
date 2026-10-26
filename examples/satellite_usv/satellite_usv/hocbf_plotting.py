#!/usr/bin/env python

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray

from .hocbf_control import HOCBFHistory
from .model import GridSpec

ROBOT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def save_hocbf_figures(
    history: HOCBFHistory,
    grid: GridSpec,
    output_dir: Path,
    movie_format: str,
    fps: int,
    movie_stride: int = 1,
) -> tuple[Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_path = output_dir / "hocbf_fields.png"
    metrics_path = output_dir / "hocbf_metrics.png"
    movie_path = output_dir / f"hocbf_evolution.{movie_format}" if movie_format != "none" else None

    _save_hocbf_fields(history, grid, fields_path)
    _save_hocbf_metrics(history, metrics_path)
    if movie_path is not None:
        _save_hocbf_animation(history, grid, movie_path, movie_format, fps, movie_stride)
    return fields_path, metrics_path, movie_path


def _save_hocbf_fields(history: HOCBFHistory, grid: GridSpec, path: Path) -> None:
    index = len(history.truth) - 1
    vmax = _state_vmax(history)
    control_vmax = max(float(np.max(control)) for control in history.control)
    source_vmax = max(float(np.max(source)) for source in history.source)
    barrier_min = min(float(np.min(barrier)) for barrier in history.barrier)
    barrier_max = max(float(np.max(barrier)) for barrier in history.barrier)
    speed = _robot_scalar_field(_velocity_infinity_norm(history.velocities[index]), history.positions[index], grid)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    panels = [
        (r"fine state $x_j^k$", history.truth[index], 0.0, vmax),
        (r"source field $q_j^k$", history.source[index], 0.0, max(source_vmax, 1e-9)),
        (r"coarse output $y_k=A x$", _expand(history.coarse_output[index], grid), 0.0, vmax),
        (r"upper reference $\bar{y}_k$", _expand(history.target_output[index], grid), 0.0, vmax),
        (r"removal rate $\beta x\kappa u$", history.control[index], 0.0, max(control_vmax, 1e-9)),
        (r"HOCBF barrier $b_k$", _expand(history.barrier[index], grid), barrier_min, barrier_max),
        (r"robot speed $\|v_i\|_\infty$", speed, 0.0, max(float(np.max(speed)), 1e-9)),
        (
            r"position barrier $h_{p,i}$",
            _robot_scalar_field(
                np.full(len(history.assigned_coarse), history.position_barrier[index]),
                history.positions[index],
                grid,
            ),
            0.0,
            1.0,
        ),
    ]
    _draw_panels(fig, axes.ravel(), panels, history, index, grid)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_hocbf_metrics(history: HOCBFHistory, path: Path) -> None:
    steps = np.arange(len(history.truth))
    actions = np.asarray(history.actions)
    speeds = np.asarray([_velocity_infinity_norm(velocity) for velocity in history.velocities])
    action_rates = np.asarray(history.action_rates)
    max_slack = [float(np.max(slack)) for slack in history.slack]
    min_hocbf = [float(np.min(residual)) for residual in history.hocbf_residual]
    min_relaxed = [
        float(np.min(residual + slack))
        for residual, slack in zip(history.hocbf_residual, history.slack)
    ]

    fig, axes = plt.subplots(10, 1, figsize=(10, 17), sharex=True, constrained_layout=True)
    axes[0].plot(steps, history.max_upper_error, label=r"$\max_k(y_k-\bar{y}_k)$")
    axes[0].axhline(history.hocbf_epsilon, color="tab:red", linestyle="--", label=r"$\varepsilon$")
    axes[0].set_ylabel("upper error")
    axes[0].legend()
    axes[1].plot(steps, [float(np.min(barrier)) for barrier in history.barrier])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel(r"$\min_k b_k$")
    axes[2].plot(steps, [float(np.min(psi1)) for psi1 in history.psi1])
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel(r"$\min_k\psi_{1,k}$")
    axes[3].plot(steps, min_hocbf, label="before slack")
    axes[3].plot(steps, min_relaxed, label="with slack")
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel(r"$\min_k\psi_{2,k}$")
    axes[3].legend()
    axes[4].plot(steps, max_slack)
    axes[4].set_ylabel(r"$\max_k\delta_k$")
    axes[5].plot(steps, history.position_barrier)
    axes[5].axhline(0.0, color="black", linewidth=0.8)
    axes[5].set_ylabel(r"$\min_i h_{p,i}$")
    for robot_index in range(actions.shape[1]):
        axes[6].plot(steps, actions[:, robot_index], label=f"i{robot_index}")
    axes[6].axhline(history.control_limit, color="tab:red", linestyle="--", alpha=0.7, label=r"$u_{\max}$")
    axes[6].set_ylabel(r"$u_i$")
    axes[6].legend(ncol=5)
    for robot_index in range(speeds.shape[1]):
        axes[7].plot(steps, speeds[:, robot_index], label=f"i{robot_index}")
    axes[7].axhline(history.velocity_limit, color="tab:red", linestyle="--", alpha=0.7, label=r"$v_{\max}$")
    axes[7].set_ylabel(r"$\|v_i\|_\infty$")
    axes[7].legend(ncol=5)
    cumulative_distance = np.asarray(history.cumulative_distance)
    for robot_index in range(cumulative_distance.shape[1]):
        axes[8].plot(steps, cumulative_distance[:, robot_index], label=f"i{robot_index}")
    axes[8].set_ylabel(r"$L_i$")
    axes[8].legend(ncol=4)
    axes[9].plot(steps, history.action_energy, label=r"$J_u=\int\sum_i u_i^2dt$")
    axes[9].plot(steps, history.velocity_energy, label=r"$J_v=\int\sum_i\|v_i\|^2dt$")
    axes[9].plot(steps, history.path_length, label=r"$L=\int\sum_i\|v_i\|dt$")
    axes[9].set_ylabel("cost")
    axes[9].set_xlabel("step")
    axes[9].legend(ncol=3)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_hocbf_animation(
    history: HOCBFHistory,
    grid: GridSpec,
    path: Path,
    movie_format: str,
    fps: int,
    movie_stride: int,
) -> None:
    frame_indices = _movie_frame_indices(len(history.truth), movie_stride)
    vmax = _state_vmax(history)
    control_vmax = max(float(np.max(control)) for control in history.control)
    source_vmax = max(float(np.max(source)) for source in history.source)
    barrier_min = min(float(np.min(barrier)) for barrier in history.barrier)
    barrier_max = max(float(np.max(barrier)) for barrier in history.barrier)
    speed_vmax = max(float(np.max(_velocity_infinity_norm(velocity))) for velocity in history.velocities)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    panels = [
        (r"fine state $x_j^k$", history.truth[0], 0.0, vmax),
        (r"source field $q_j^k$", history.source[0], 0.0, max(source_vmax, 1e-9)),
        (r"coarse output $y_k=A x$", _expand(history.coarse_output[0], grid), 0.0, vmax),
        (r"upper reference $\bar{y}_k$", _expand(history.target_output[0], grid), 0.0, vmax),
        (r"removal rate $\beta x\kappa u$", history.control[0], 0.0, max(control_vmax, 1e-9)),
        (r"HOCBF barrier $b_k$", _expand(history.barrier[0], grid), barrier_min, barrier_max),
        (
            r"robot speed $\|v_i\|_\infty$",
            _robot_scalar_field(_velocity_infinity_norm(history.velocities[0]), history.positions[0], grid),
            0.0,
            max(speed_vmax, 1e-9),
        ),
        (
            r"position barrier $h_{p,i}$",
            _robot_scalar_field(
                np.full(len(history.assigned_coarse), history.position_barrier[0]),
                history.positions[0],
                grid,
            ),
            0.0,
            1.0,
        ),
    ]
    images, overlays = _draw_panels(fig, axes.ravel(), panels, history, 0, grid, animated=True)

    def update(frame_index: int) -> list:
        images[0].set_data(history.truth[frame_index])
        images[1].set_data(history.source[frame_index])
        images[2].set_data(_expand(history.coarse_output[frame_index], grid))
        images[3].set_data(_expand(history.target_output[frame_index], grid))
        images[4].set_data(history.control[frame_index])
        images[5].set_data(_expand(history.barrier[frame_index], grid))
        speed_field = _robot_scalar_field(
            _velocity_infinity_norm(history.velocities[frame_index]),
            history.positions[frame_index],
            grid,
        )
        images[6].set_data(speed_field)
        position_barrier_field = _robot_scalar_field(
            np.full(len(history.assigned_coarse), history.position_barrier[frame_index]),
            history.positions[frame_index],
            grid,
        )
        images[7].set_data(position_barrier_field)
        _update_robot_overlays(overlays, history, frame_index)
        fig.suptitle(
            rf"step {frame_index:03d}    $\min b_k$={float(np.min(history.barrier[frame_index])):.4f}    "
            rf"$\max\delta_k$={float(np.max(history.slack[frame_index])):.2e}"
        )
        artists = [*images]
        for overlay in overlays:
            artists.extend(overlay["current"])
            artists.extend(overlay["targets"])
            artists.extend(overlay["paths"])
        return artists

    animation = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps) if movie_format == "gif" else FFMpegWriter(fps=fps)
    animation.save(path, writer=writer, dpi=120)
    plt.close(fig)


def _draw_panels(
    fig: plt.Figure,
    axes: np.ndarray,
    panels: list[tuple[str, np.ndarray, float, float]],
    history: HOCBFHistory,
    frame_index: int,
    grid: GridSpec,
    animated: bool = False,
) -> tuple[list, list[dict[str, list]]]:
    images = []
    overlays = []
    for axis_index, (axis, (title, data, vmin, vmax)) in enumerate(zip(axes, panels)):
        image = axis.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="turbo", animated=animated)
        axis.set_title(title)
        _style_axis(axis, grid)
        overlay = _draw_robot_overlay(axis, history, frame_index, animated)
        if axis_index == 0:
            _add_robot_legend(axis)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        images.append(image)
        overlays.append(overlay)
    return images, overlays


def _draw_robot_overlay(
    axis: plt.Axes,
    history: HOCBFHistory,
    frame_index: int,
    animated: bool,
) -> dict[str, list]:
    current_markers = []
    target_markers = []
    path_lines = []
    start_positions = history.positions[0]
    current_positions = history.positions[frame_index]
    current_targets = history.nominal_targets[frame_index]
    positions = np.asarray(history.positions[: frame_index + 1])

    for robot_index, color in enumerate(ROBOT_COLORS[: current_positions.shape[0]]):
        path_line = axis.plot(
            positions[:, robot_index, 1],
            positions[:, robot_index, 0],
            color=color,
            linewidth=1.5,
            alpha=0.9,
            animated=animated,
        )[0]
        start_marker = axis.plot(
            start_positions[robot_index, 1],
            start_positions[robot_index, 0],
            marker="s",
            color=color,
            markeredgecolor="black",
            linestyle="",
            markersize=6,
            alpha=0.9,
            animated=animated,
        )[0]
        target_marker = axis.plot(
            current_targets[robot_index, 1],
            current_targets[robot_index, 0],
            marker="x",
            color=color,
            linestyle="",
            markersize=8,
            markeredgewidth=2.0,
            animated=animated,
        )[0]
        current_marker = axis.plot(
            current_positions[robot_index, 1],
            current_positions[robot_index, 0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="",
            markersize=8,
            animated=animated,
        )[0]
        path_lines.append(path_line)
        target_markers.append(target_marker)
        current_markers.append(current_marker)

    return {
        "current": current_markers,
        "targets": target_markers,
        "paths": path_lines,
        "starts": [start_marker],
    }


def _add_robot_legend(axis: plt.Axes) -> None:
    handles = [
        Line2D([0], [0], color="tab:blue", linewidth=1.5, label="trajectory"),
        Line2D([0], [0], marker="s", color="tab:blue", linestyle="", markeredgecolor="black", label="initial"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="", label="current"),
        Line2D([0], [0], marker="x", color="tab:blue", linestyle="", markeredgewidth=2.0, label="nominal target"),
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)


def _update_robot_overlays(overlays: list[dict[str, list]], history: HOCBFHistory, frame_index: int) -> None:
    positions = np.asarray(history.positions[: frame_index + 1])
    current_positions = history.positions[frame_index]
    current_targets = history.nominal_targets[frame_index]
    for overlay in overlays:
        for robot_index, path_line in enumerate(overlay["paths"]):
            path_line.set_data(positions[:, robot_index, 1], positions[:, robot_index, 0])
        for robot_index, target_marker in enumerate(overlay["targets"]):
            target_marker.set_data([current_targets[robot_index, 1]], [current_targets[robot_index, 0]])
        for robot_index, current_marker in enumerate(overlay["current"]):
            current_marker.set_data([current_positions[robot_index, 1]], [current_positions[robot_index, 0]])


def _robot_scalar_field(
    values: NDArray[np.float64],
    positions: NDArray[np.float64],
    grid: GridSpec,
) -> NDArray[np.float64]:
    field = np.zeros(grid.fine_shape, dtype=np.float64)
    for value, position in zip(values, positions):
        row = int(np.clip(round(float(position[0])), 0, grid.fine_shape[0] - 1))
        col = int(np.clip(round(float(position[1])), 0, grid.fine_shape[1] - 1))
        field[row, col] = float(value)
    return field


def _velocity_infinity_norm(velocities: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.max(np.abs(velocities), axis=1)


def _state_vmax(history: HOCBFHistory) -> float:
    return max(
        max(float(np.max(field)) for field in history.truth),
        max(float(np.max(output)) for output in history.coarse_output),
        max(float(np.max(target)) for target in history.target_output),
        1e-9,
    )


def _movie_frame_indices(frame_count: int, stride: int) -> list[int]:
    stride = max(int(stride), 1)
    indices = list(range(0, frame_count, stride))
    last_index = frame_count - 1
    if indices[-1] != last_index:
        indices.append(last_index)
    return indices


def _expand(coarse: np.ndarray, grid: GridSpec) -> np.ndarray:
    return np.kron(coarse.reshape(grid.coarse_shape), np.ones(grid.fine_per_coarse))


def _style_axis(axis: plt.Axes, grid: GridSpec) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    rows_per_block, cols_per_block = grid.fine_per_coarse
    for row in range(1, grid.fine_shape[0]):
        axis.axhline(row - 0.5, color="black", linewidth=0.35, alpha=0.18)
    for col in range(1, grid.fine_shape[1]):
        axis.axvline(col - 0.5, color="black", linewidth=0.35, alpha=0.18)
    for row in range(rows_per_block, grid.fine_shape[0], rows_per_block):
        axis.axhline(row - 0.5, color="white", linewidth=1.4, alpha=0.95)
    for col in range(cols_per_block, grid.fine_shape[1], cols_per_block):
        axis.axvline(col - 0.5, color="white", linewidth=1.4, alpha=0.95)
