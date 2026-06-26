#!/usr/bin/env python

import argparse
from pathlib import Path

from .control import ControlConfig, run_control_comparison, run_control_simulation
from .control_plotting import save_control_figures
from .model import GridSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the hierarchical environment-control demo.")
    parser.add_argument("--steps", type=int, default=400, help="number of simulation steps")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--planner",
        choices=["no_control", "random_control", "greedy_fine", "hierarchical"],
        default="hierarchical",
        help="control planner",
    )
    parser.add_argument("--usv-count", type=int, default=3, help="number of USVs")
    parser.add_argument("--coarse-size", type=int, default=4, help="coarse grid side length")
    parser.add_argument("--fine-per-coarse", type=int, default=4, help="fine cells per coarse-cell side")
    parser.add_argument("--target-scale", type=float, default=0.82, help="scale of the fixed coarse target map")
    parser.add_argument(
        "--control-amount",
        type=float,
        default=1.0,
        help="local control amount applied at the USV cell",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("satellite_usv_control_demo"))
    parser.add_argument("--movie-format", choices=["gif", "mp4", "none"], default="gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--movie-stride", type=int, default=4, help="save one movie frame every N simulation steps")
    parser.add_argument("--comparison", action="store_true", help="also run baseline planners")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid = GridSpec(
        coarse_shape=(args.coarse_size, args.coarse_size),
        fine_per_coarse=(args.fine_per_coarse, args.fine_per_coarse),
    )
    config = ControlConfig(
        steps=args.steps,
        seed=args.seed,
        planner=args.planner,
        usv_count=args.usv_count,
        target_scale=args.target_scale,
        control_amount=args.control_amount,
    )
    history = run_control_simulation(config, grid)
    comparison = run_control_comparison(config, grid) if args.comparison else None
    fields_path, metrics_path, comparison_path, movie_path = save_control_figures(
        history,
        comparison,
        grid,
        args.output_dir,
        args.movie_format,
        args.fps,
        args.movie_stride,
    )

    print("satellite-USV control demo completed")
    print(f"planner: {config.planner}")
    print(f"coarse grid: {grid.coarse_shape}, fine grid: {grid.fine_shape}")
    print(f"steps: {config.steps}")
    print(f"USVs: {config.usv_count}")
    print(f"movie stride: {args.movie_stride}")
    print(f"final tracking error: {history.tracking_error[-1]:.3f}")
    print(f"final total concentration: {history.total_concentration[-1]:.3f}")
    print(f"final max concentration: {history.max_concentration[-1]:.3f}")
    print(f"fields figure: {fields_path}")
    print(f"metrics figure: {metrics_path}")
    if comparison_path is not None:
        print(f"comparison figure: {comparison_path}")
    if movie_path is not None:
        print(f"movie: {movie_path}")


if __name__ == "__main__":
    main()
