#!/usr/bin/env python

import argparse
from pathlib import Path

from .model import GridSpec, SimulationConfig, run_simulation
from .plotting import save_animation, save_mode_comparison, save_summary_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal satellite-to-USV coverage simulation.")
    parser.add_argument("--steps", type=int, default=40, help="number of simulation steps")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--mode",
        choices=["none", "ideal", "usv"],
        default="none",
        help="control mode: none for observation-only, ideal all-cell control, or local USV control",
    )
    parser.add_argument(
        "--planner",
        choices=["tracking", "satellite_survey", "greedy"],
        default="tracking",
        help="USV motion planner",
    )
    parser.add_argument("--usv-count", type=int, default=2, help="number of USVs for local sensing/control")
    parser.add_argument("--coarse-size", type=int, default=5, help="coarse satellite grid side length")
    parser.add_argument("--fine-per-coarse", type=int, default=3, help="fine cells per coarse-cell side")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("satellite_usv_demo"),
        help="directory where figures and movies are written",
    )
    parser.add_argument(
        "--movie-format",
        choices=["gif", "mp4", "none"],
        default="gif",
        help="movie output format",
    )
    parser.add_argument("--fps", type=int, default=6, help="movie frames per second")
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="also write a none/ideal/usv control comparison figure",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid = GridSpec(
        coarse_shape=(args.coarse_size, args.coarse_size),
        fine_per_coarse=(args.fine_per_coarse, args.fine_per_coarse),
    )
    config = SimulationConfig(
        steps=args.steps,
        seed=args.seed,
        control_mode=args.mode,
        planner_mode=args.planner,
        usv_count=args.usv_count,
    )
    history = run_simulation(config=config, grid=grid)
    fields_path, metrics_path, output_path = save_summary_figures(history, grid, args.output_dir)
    movie_path = None
    if args.movie_format != "none":
        movie_path = save_animation(history, grid, args.output_dir, args.movie_format, args.fps)

    comparison_path = None
    if args.comparison:
        histories = {
            mode: run_simulation(
                config=SimulationConfig(
                    steps=args.steps,
                    seed=args.seed,
                    control_mode=mode,
                    planner_mode=args.planner,
                    usv_count=args.usv_count,
                ),
                grid=grid,
            )
            for mode in ["none", "ideal", "usv"]
        }
        comparison_path = save_mode_comparison(histories, grid, args.output_dir)

    print("satellite-USV demo completed")
    print(f"mode: {config.control_mode}")
    print(f"planner: {config.planner_mode}")
    print(f"coarse grid: {grid.coarse_shape}, fine grid: {grid.fine_shape}")
    print(f"steps: {config.steps}")
    print(f"final total concentration: {history.total_concentration[-1]:.3f}")
    print(f"final fine RMSE: {history.fine_rmse[-1]:.3f}")
    print(f"final weighted RMSE: {history.weighted_rmse[-1]:.3f}")
    print(f"final output tracking error: {history.tracking_error[-1]:.3f}")
    print(f"observed fine-cell fraction: {history.observed_fraction[-1]:.3f}")
    print(f"fields figure: {fields_path}")
    print(f"metrics figure: {metrics_path}")
    print(f"output tracking figure: {output_path}")
    if movie_path is not None:
        print(f"movie: {movie_path}")
    if comparison_path is not None:
        print(f"mode comparison: {comparison_path}")


if __name__ == "__main__":
    main()
