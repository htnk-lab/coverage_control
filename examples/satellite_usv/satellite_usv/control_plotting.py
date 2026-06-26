#!/usr/bin/env python

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np

from .control import ControlHistory
from .model import GridSpec


def save_control_figures(
    history: ControlHistory,
    comparison: dict[str, ControlHistory] | None,
    grid: GridSpec,
    output_dir: Path,
    movie_format: str,
    fps: int,
    movie_stride: int = 1,
) -> tuple[Path, Path, Path | None, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_path = output_dir / "control_fields.png"
    metrics_path = output_dir / "control_metrics.png"
    comparison_path = output_dir / "control_comparison.png" if comparison is not None else None
    movie_path = output_dir / f"control_evolution.{movie_format}" if movie_format != "none" else None

    _save_control_fields(history, grid, fields_path)
    _save_control_metrics(history, metrics_path)
    if comparison is not None and comparison_path is not None:
        _save_control_comparison(comparison, grid, comparison_path)
    if movie_path is not None:
        _save_control_animation(history, grid, movie_path, movie_format, fps, movie_stride)
    return fields_path, metrics_path, comparison_path, movie_path


def _save_control_fields(history: ControlHistory, grid: GridSpec, path: Path) -> None:
    truth = history.truth[-1]
    coarse_output = _expand(history.coarse_output[-1], grid)
    target_output = _expand(history.target_output[-1], grid)
    reduction = np.maximum(history.truth[0] - truth, 0.0)
    vmax = max(float(np.max(truth)), float(np.max(coarse_output)), float(np.max(target_output)), 1e-9)
    reduction_vmax = max(float(np.max(reduction)), 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    panels = [
        (r"fine environmental state $x_k$", truth, 0.0, vmax),
        (r"coarse output $y_k = A x_k$", coarse_output, 0.0, vmax),
        (r"fixed upper target $\bar{y}$", target_output, 0.0, vmax),
        (r"reduction from initial $x_0 - x_k$", reduction, 0.0, reduction_vmax),
    ]
    for axis, (title, data, vmin, local_vmax) in zip(axes.ravel(), panels):
        image = axis.imshow(data, origin="lower", vmin=vmin, vmax=local_vmax, cmap="turbo")
        axis.set_title(title)
        _style_axis(axis, grid)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for position in history.usv_positions[-1]:
        for axis in axes.ravel():
            axis.plot(position[1], position[0], marker="o", color="white", markeredgecolor="black", markersize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_control_metrics(history: ControlHistory, path: Path) -> None:
    steps = np.arange(len(history.truth))
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    axes[0].plot(steps, history.tracking_error)
    axes[0].set_ylabel(r"$\|y-\bar{y}\|$")
    axes[1].plot(steps, history.total_concentration)
    axes[1].set_ylabel(r"$\sum_i x_i$")
    axes[2].plot(steps, history.max_concentration)
    axes[2].set_ylabel(r"$\max_i x_i$")
    axes[3].plot(steps, history.control_cost)
    axes[3].set_ylabel(r"$\sum_r a_r^2$")
    axes[3].set_xlabel("step")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_control_comparison(histories: dict[str, ControlHistory], grid: GridSpec, path: Path) -> None:
    colors = {
        "no_control": "tab:gray",
        "random_control": "tab:orange",
        "greedy_fine": "tab:blue",
        "hierarchical": "tab:green",
    }
    labels = {
        "no_control": "no control",
        "random_control": "random USV control",
        "greedy_fine": "greedy fine control",
        "hierarchical": "hierarchical control",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for mode, history in histories.items():
        steps = np.arange(len(history.truth))
        color = colors.get(mode)
        label = labels.get(mode, mode)
        axes[0, 0].plot(steps, history.tracking_error, label=label, color=color)
        axes[0, 1].plot(steps, history.total_concentration, label=label, color=color)
        axes[0, 2].plot(steps, history.max_concentration, label=label, color=color)

    axes[0, 0].set_title(r"coarse tracking error $\|y-\bar{y}\|$")
    axes[0, 1].set_title(r"total environmental state $\sum_i x_i$")
    axes[0, 2].set_title(r"hotspot magnitude $\max_i x_i$")
    for axis in axes[0]:
        axis.grid(True, alpha=0.3)
        axis.legend()

    vmax = max(float(np.max(history.truth[-1])) for history in histories.values())
    for axis, mode in zip(axes[1], ["no_control", "greedy_fine", "hierarchical"]):
        history = histories[mode]
        image = axis.imshow(history.truth[-1], origin="lower", vmin=0.0, vmax=vmax, cmap="turbo")
        axis.set_title(f"final $x_k$: {labels.get(mode, mode)}")
        _style_axis(axis, grid)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_control_animation(
    history: ControlHistory,
    grid: GridSpec,
    path: Path,
    movie_format: str,
    fps: int,
    movie_stride: int,
) -> None:
    frame_indices = _movie_frame_indices(len(history.truth), movie_stride)
    vmax = max(float(np.max(field)) for field in history.truth)
    reduction_vmax = max(float(np.max(np.maximum(history.truth[0] - field, 0.0))) for field in history.truth)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    images = [
        axes[0, 0].imshow(history.truth[0], origin="lower", vmin=0.0, vmax=vmax, cmap="turbo", animated=True),
        axes[0, 1].imshow(
            _expand(history.coarse_output[0], grid), origin="lower", vmin=0.0, vmax=vmax, cmap="turbo", animated=True
        ),
        axes[1, 0].imshow(
            _expand(history.target_output[0], grid), origin="lower", vmin=0.0, vmax=vmax, cmap="turbo", animated=True
        ),
        axes[1, 1].imshow(
            np.maximum(history.truth[0] - history.truth[0], 0.0),
            origin="lower",
            vmin=0.0,
            vmax=reduction_vmax,
            cmap="turbo",
            animated=True,
        ),
    ]
    titles = [
        r"fine environmental state $x_k$",
        r"coarse output $y_k = A x_k$",
        r"fixed upper target $\bar{y}$",
        r"reduction from initial $x_0-x_k$",
    ]
    markers = []
    for axis, image, title in zip(axes.ravel(), images, titles):
        axis.set_title(title)
        _style_axis(axis, grid)
        markers.append(axis.plot([], [], marker="o", color="white", markeredgecolor="black", linestyle="")[0])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    def update(frame_index: int) -> list:
        images[0].set_data(history.truth[frame_index])
        images[1].set_data(_expand(history.coarse_output[frame_index], grid))
        images[2].set_data(_expand(history.target_output[frame_index], grid))
        images[3].set_data(np.maximum(history.truth[0] - history.truth[frame_index], 0.0))
        rows = [position[0] for position in history.usv_positions[frame_index]]
        cols = [position[1] for position in history.usv_positions[frame_index]]
        for marker in markers:
            marker.set_data(cols, rows)
        fig.suptitle(
            rf"step {frame_index:03d}    $\|y-\bar{{y}}\|$={history.tracking_error[frame_index]:.3f}    "
            rf"$\sum_i x_i$={history.total_concentration[frame_index]:.2f}"
        )
        return [*images, *markers]

    animation = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps) if movie_format == "gif" else FFMpegWriter(fps=fps)
    animation.save(path, writer=writer, dpi=120)
    plt.close(fig)


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
    for row in range(rows_per_block, grid.fine_shape[0], rows_per_block):
        axis.axhline(row - 0.5, color="white", linewidth=1.0, alpha=0.8)
    for col in range(cols_per_block, grid.fine_shape[1], cols_per_block):
        axis.axvline(col - 0.5, color="white", linewidth=1.0, alpha=0.8)
