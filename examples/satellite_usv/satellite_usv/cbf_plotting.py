#!/usr/bin/env python

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from .cbf_control import CBFHistory
from .model import GridSpec


def save_cbf_figures(
    history: CBFHistory,
    grid: GridSpec,
    output_dir: Path,
    movie_format: str,
    fps: int,
    movie_stride: int = 1,
) -> tuple[Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_path = output_dir / "cbf_fields.png"
    metrics_path = output_dir / "cbf_metrics.png"
    movie_path = output_dir / f"cbf_evolution.{movie_format}" if movie_format != "none" else None

    _save_cbf_fields(history, grid, fields_path)
    _save_cbf_metrics(history, metrics_path)
    if movie_path is not None:
        _save_cbf_animation(history, grid, movie_path, movie_format, fps, movie_stride)
    return fields_path, metrics_path, movie_path


def _save_cbf_fields(history: CBFHistory, grid: GridSpec, path: Path) -> None:
    final_index = len(history.truth) - 1
    vmax = _state_vmax(history)
    control_vmax = max(float(np.max(control)) for control in history.control)
    barrier_min = min(float(np.min(barrier)) for barrier in history.barrier)
    barrier_max = max(float(np.max(barrier)) for barrier in history.barrier)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    panels = [
        (r"fine state $x_k$", history.truth[final_index], 0.0, vmax),
        (r"coarse output $y_k=A x_k$", _expand(history.coarse_output[final_index], grid), 0.0, vmax),
        (r"upper reference $\bar{y}_k$", _expand(history.target_output[final_index], grid), 0.0, vmax),
        (r"desired output $y^{\mathrm{des}}$", _expand(history.desired_output[final_index], grid), 0.0, vmax),
        (r"local control field $u_k$", history.control[final_index], 0.0, max(control_vmax, 1e-9)),
        (r"CBF barrier $b_i$", _expand(history.barrier[final_index], grid), barrier_min, barrier_max),
    ]
    _draw_panels(fig, axes.ravel(), panels, history.usv_positions[final_index], grid)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_cbf_metrics(history: CBFHistory, path: Path) -> None:
    steps = np.arange(len(history.truth))
    cbf_inputs = np.asarray(history.cbf_input)
    relaxed_margin = [
        float(np.min(margin + slack))
        for margin, slack in zip(history.cbf_margin, history.slack)
    ]
    max_slack = [float(np.max(slack)) for slack in history.slack]
    min_discrete_residual = [float(np.min(residual)) for residual in history.discrete_cbf_residual]
    epsilon = history.cbf_epsilon
    input_limit = history.control_limit

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    axes[0].plot(steps, history.max_abs_error, label=r"$\max_i |e_i|$")
    axes[0].axhline(epsilon, color="tab:red", linestyle="--", label=r"$\varepsilon$")
    axes[0].set_ylabel("cell error")
    axes[0].legend()
    axes[1].plot(steps, [float(np.min(barrier)) for barrier in history.barrier])
    axes[1].set_ylabel(r"$\min_i b_i$")
    axes[2].plot(steps, [float(np.min(margin)) for margin in history.cbf_margin], label="before slack")
    axes[2].plot(steps, relaxed_margin, label="with slack")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel(r"$\min_i(\dot b_i+\gamma b_i)$")
    axes[2].legend()
    axes[3].plot(steps, max_slack)
    axes[3].set_ylabel(r"$\max_i \delta_i$")
    axes[4].plot(steps, min_discrete_residual)
    axes[4].axhline(0.0, color="black", linewidth=0.8)
    axes[4].set_ylabel(r"$\min_i \Delta b_i^{disc}$")
    for coarse_flat in range(cbf_inputs.shape[1] * cbf_inputs.shape[2]):
        coarse_index = np.unravel_index(coarse_flat, cbf_inputs.shape[1:])
        axes[5].plot(steps, cbf_inputs[:, coarse_index[0], coarse_index[1]], label=f"{coarse_index}")
    axes[5].axhline(input_limit, color="tab:red", linestyle="--", alpha=0.7, label=r"$a_{\max}$")
    axes[5].set_ylabel(r"$a_r$")
    axes[5].set_xlabel("step")
    axes[5].legend(title="assigned coarse cell", ncol=5)
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_cbf_animation(
    history: CBFHistory,
    grid: GridSpec,
    path: Path,
    movie_format: str,
    fps: int,
    movie_stride: int,
) -> None:
    frame_indices = _movie_frame_indices(len(history.truth), movie_stride)
    vmax = _state_vmax(history)
    control_vmax = max(float(np.max(control)) for control in history.control)
    barrier_min = min(float(np.min(barrier)) for barrier in history.barrier)
    barrier_max = max(float(np.max(barrier)) for barrier in history.barrier)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    initial_panels = [
        (r"fine state $x_k$", history.truth[0], 0.0, vmax),
        (r"coarse output $y_k=A x_k$", _expand(history.coarse_output[0], grid), 0.0, vmax),
        (r"upper reference $\bar{y}_k$", _expand(history.target_output[0], grid), 0.0, vmax),
        (r"desired output $y^{\mathrm{des}}$", _expand(history.desired_output[0], grid), 0.0, vmax),
        (r"local control field $u_k$", history.control[0], 0.0, max(control_vmax, 1e-9)),
        (r"CBF barrier $b_i$", _expand(history.barrier[0], grid), barrier_min, barrier_max),
    ]
    images, markers = _draw_panels(fig, axes.ravel(), initial_panels, history.usv_positions[0], grid, animated=True)

    def update(frame_index: int) -> list:
        images[0].set_data(history.truth[frame_index])
        images[1].set_data(_expand(history.coarse_output[frame_index], grid))
        images[2].set_data(_expand(history.target_output[frame_index], grid))
        images[3].set_data(_expand(history.desired_output[frame_index], grid))
        images[4].set_data(history.control[frame_index])
        images[5].set_data(_expand(history.barrier[frame_index], grid))
        rows = [position[0] for position in history.usv_positions[frame_index]]
        cols = [position[1] for position in history.usv_positions[frame_index]]
        for marker in markers:
            marker.set_data(cols, rows)
        fig.suptitle(
            rf"step {frame_index:03d}    $\|y-\bar{{y}}\|$={history.tracking_error[frame_index]:.3f}    "
            rf"$\min b_i$={float(np.min(history.barrier[frame_index])):.4f}"
        )
        return [*images, *markers]

    animation = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps) if movie_format == "gif" else FFMpegWriter(fps=fps)
    animation.save(path, writer=writer, dpi=120)
    plt.close(fig)


def _draw_panels(
    fig: plt.Figure,
    axes: np.ndarray,
    panels: list[tuple[str, np.ndarray, float, float]],
    usv_positions: list[tuple[int, int]],
    grid: GridSpec,
    animated: bool = False,
) -> tuple[list, list]:
    images = []
    markers = []
    for axis, (title, data, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(data, origin="lower", vmin=vmin, vmax=vmax, cmap="turbo", animated=animated)
        axis.set_title(title)
        _style_axis(axis, grid)
        marker = axis.plot(
            [position[1] for position in usv_positions],
            [position[0] for position in usv_positions],
            marker="o",
            color="white",
            markeredgecolor="black",
            linestyle="",
            markersize=8,
        )[0]
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        images.append(image)
        markers.append(marker)
    return images, markers


def _state_vmax(history: CBFHistory) -> float:
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
