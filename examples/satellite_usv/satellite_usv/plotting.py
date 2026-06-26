#!/usr/bin/env python

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from .model import GridSpec, SimulationHistory


def save_summary_figures(history: SimulationHistory, grid: GridSpec, output_dir: Path) -> tuple[Path, Path, Path]:
    """Save compact PNG summaries for the toy simulation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_path = output_dir / "satellite_usv_fields.png"
    metrics_path = output_dir / "satellite_usv_metrics.png"
    output_path = output_dir / "satellite_usv_output_tracking.png"

    truth = history.truth[-1]
    estimate = history.estimate[-1]
    y_ref_expanded = _expand_for_display(history.y_ref[-1], grid)
    y_output_expanded = _expand_for_display(history.y_output[-1], grid)

    vmax = float(max(np.max(truth), np.max(estimate), 1e-9))
    output_vmax = float(max(np.max(y_ref_expanded), np.max(y_output_expanded), 1e-9))
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panels = [
        (r"fine truth $x_k$", truth, vmax),
        (r"desired observation output $y_{\rm ref}$", y_ref_expanded, output_vmax),
        (r"estimate $\hat{x}_k$", estimate, vmax),
        (r"USV observation output $y_k = H z_k$", y_output_expanded, output_vmax),
    ]
    for axis, (title, data, local_vmax) in zip(axes.ravel(), panels):
        image = axis.imshow(data, origin="lower", vmin=0.0, vmax=local_vmax, cmap="turbo")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        _draw_coarse_grid(axis, grid)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for position in history.usv_positions[-1]:
        for axis in axes.ravel()[:3]:
            axis.plot(position[1], position[0], marker="o", color="white", markeredgecolor="black", markersize=8)

    fig.savefig(fields_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    steps = np.arange(len(history.total_concentration))
    axes[0].plot(steps, history.total_concentration)
    axes[0].set_ylabel(r"$\sum_i x_i$")
    axes[1].plot(steps, history.tracking_error)
    axes[1].set_ylabel(r"$\|y-y_{\rm ref}\|$")
    axes[2].plot(steps, history.fine_rmse, label="RMSE")
    axes[2].plot(steps, history.weighted_rmse, label="weighted")
    axes[2].set_ylabel(r"$\hat{x}$ error")
    axes[3].plot(steps, history.observed_fraction)
    axes[3].set_ylabel("observed")
    axes[3].set_xlabel("step")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[2].legend()
    fig.savefig(metrics_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    diff = history.y_output[-1] - history.y_ref[-1]
    diff_absmax = float(max(np.max(np.abs(diff)), 1e-9))
    output_panels = [
        (r"desired $y_{\rm ref}$", history.y_ref[-1], output_vmax),
        (r"actual $y_k = H z_k$", history.y_output[-1], output_vmax),
        (r"actual - desired", diff, diff_absmax),
    ]
    for axis, (title, data, local_vmax) in zip(axes, output_panels):
        if title == r"actual - desired":
            image = axis.imshow(data, origin="lower", vmin=-local_vmax, vmax=local_vmax, cmap="turbo")
        else:
            image = axis.imshow(data, origin="lower", vmin=0.0, vmax=local_vmax, cmap="turbo")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return fields_path, metrics_path, output_path


def save_animation(
    history: SimulationHistory,
    grid: GridSpec,
    output_dir: Path,
    movie_format: str = "gif",
    fps: int = 6,
) -> Path:
    """Save a time-evolution movie for the satellite-USV simulation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    movie_format = movie_format.lower()
    if movie_format not in {"gif", "mp4"}:
        raise ValueError("movie_format must be 'gif' or 'mp4'")

    movie_path = output_dir / f"satellite_usv_evolution.{movie_format}"
    vmax = _history_field_vmax(history, grid)
    output_vmax = _history_output_vmax(history, grid)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panel_specs = [
        (r"fine truth $x_k$", history.truth[0], vmax),
        (r"desired observation output $y_{\rm ref}$", _expand_for_display(history.y_ref[0], grid), output_vmax),
        (r"estimate $\hat{x}_k$", history.estimate[0], vmax),
        (r"USV observation output $y_k = H z_k$", _expand_for_display(history.y_output[0], grid), output_vmax),
    ]
    images = []
    markers = []
    for axis, (title, data, local_vmax) in zip(axes.ravel(), panel_specs):
        image = axis.imshow(data, origin="lower", vmin=0.0, vmax=local_vmax, cmap="turbo", animated=True)
        images.append(image)
        axis.set_title(title)
        _style_grid_axis(axis, grid)
        marker = axis.plot([], [], marker="o", color="white", markeredgecolor="black", markersize=8, linestyle="")[0]
        markers.append(marker)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(_frame_title(history, 0))

    def update(frame_index: int) -> list:
        images[0].set_data(history.truth[frame_index])
        images[1].set_data(_expand_for_display(history.y_ref[frame_index], grid))
        images[2].set_data(history.estimate[frame_index])
        images[3].set_data(_expand_for_display(history.y_output[frame_index], grid))

        rows = [position[0] for position in history.usv_positions[frame_index]]
        cols = [position[1] for position in history.usv_positions[frame_index]]
        for marker in markers[:3]:
            marker.set_data(cols, rows)
        markers[3].set_data([], [])
        fig.suptitle(_frame_title(history, frame_index))
        return [*images, *markers]

    animation = FuncAnimation(fig, update, frames=len(history.truth), interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps) if movie_format == "gif" else FFMpegWriter(fps=fps)
    animation.save(movie_path, writer=writer, dpi=120)
    plt.close(fig)
    return movie_path


def save_mode_comparison(histories: dict[str, SimulationHistory], grid: GridSpec, output_dir: Path) -> Path:
    """Compare no-control, ideal-control, and USV-local-control runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "satellite_usv_mode_comparison.png"
    colors = {"none": "tab:gray", "ideal": "tab:blue", "usv": "tab:green"}
    labels = {
        "none": "observation only",
        "ideal": "ideal full-field actuation",
        "usv": "USV-local actuation",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    metric_axes = axes[0]
    final_axes = axes[1]

    for mode, history in histories.items():
        steps = np.arange(len(history.total_concentration))
        color = colors.get(mode, None)
        label = labels.get(mode, mode)
        metric_axes[0].plot(steps, history.total_concentration, label=label, color=color)
        metric_axes[1].plot(steps, [float(np.sum(control)) for control in history.control], label=label, color=color)
        metric_axes[2].plot(steps, history.fine_rmse, label=label, color=color)

    metric_axes[0].set_title(r"total environmental state $\sum_i x_i$")
    metric_axes[0].set_ylabel(r"$\sum_i x_i$")
    metric_axes[1].set_title(r"diagnostic actuation effort $\sum_i u_i$")
    metric_axes[1].set_ylabel(r"$\sum_i u_i$")
    metric_axes[2].set_title(r"estimation error $\|\hat{x}-x\|$")
    metric_axes[2].set_ylabel("fine RMSE")
    for axis in metric_axes:
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.3)
        axis.legend()

    vmax = max(float(np.max(history.truth[-1])) for history in histories.values())
    for axis, mode in zip(final_axes, ["none", "ideal", "usv"]):
        history = histories[mode]
        image = axis.imshow(history.truth[-1], origin="lower", vmin=0.0, vmax=vmax, cmap="turbo")
        axis.set_title(f"final $x_k$: {labels.get(mode, mode)}")
        _style_grid_axis(axis, grid)
        for position in history.usv_positions[-1]:
            axis.plot(position[1], position[0], marker="o", color="white", markeredgecolor="black", markersize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.savefig(comparison_path, dpi=160)
    plt.close(fig)
    return comparison_path


def _expand_for_display(coarse: np.ndarray, grid: GridSpec) -> np.ndarray:
    return np.kron(coarse.reshape(grid.coarse_shape), np.ones(grid.fine_per_coarse))


def _history_field_vmax(history: SimulationHistory, grid: GridSpec) -> float:
    truth_vmax = max(float(np.max(field)) for field in history.truth)
    estimate_vmax = max(float(np.max(field)) for field in history.estimate)
    return max(truth_vmax, estimate_vmax, 1e-9)


def _history_output_vmax(history: SimulationHistory, grid: GridSpec) -> float:
    y_ref_vmax = max(float(np.max(_expand_for_display(field, grid))) for field in history.y_ref)
    y_output_vmax = max(float(np.max(_expand_for_display(field, grid))) for field in history.y_output)
    return max(y_ref_vmax, y_output_vmax, 1e-9)


def _frame_title(history: SimulationHistory, frame_index: int) -> str:
    rmse = history.fine_rmse[frame_index]
    tracking_error = history.tracking_error[frame_index]
    observed = history.observed_fraction[frame_index]
    return (
        rf"observation demo    step {frame_index:03d}    "
        rf"$\|y-y_{{\rm ref}}\|$={tracking_error:.3f}    RMSE={rmse:.3f}    observed={observed:.2f}"
    )


def _style_grid_axis(axis: plt.Axes, grid: GridSpec) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    _draw_coarse_grid(axis, grid)


def _draw_coarse_grid(axis: plt.Axes, grid: GridSpec) -> None:
    rows_per_block, cols_per_block = grid.fine_per_coarse
    for row in range(rows_per_block, grid.fine_shape[0], rows_per_block):
        axis.axhline(row - 0.5, color="white", linewidth=1.0, alpha=0.8)
    for col in range(cols_per_block, grid.fine_shape[1], cols_per_block):
        axis.axvline(col - 0.5, color="white", linewidth=1.0, alpha=0.8)
