#!/usr/bin/env python

import argparse
from pathlib import Path

from .hocbf_control import HOCBFConfig, run_hocbf_simulation
from .hocbf_plotting import save_hocbf_figures
from .model import GridSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal HOCBF toy demo.")
    parser.add_argument("--steps", type=int, default=240, help="number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.05, help="time step")
    parser.add_argument("--target-rate", type=float, default=0.35, help="decay rate of the upper reference")
    parser.add_argument("--velocity-limit", type=float, default=1.4, help="component-wise robot velocity bound")
    parser.add_argument("--control-limit", type=float, default=1.2, help="upper bound for intervention intensity u")
    parser.add_argument("--w-limit", type=float, default=2.5, help="component-wise intervention-rate nu bound")
    parser.add_argument(
        "--source-mode",
        choices=["static", "moving", "none"],
        default="static",
        help="source field mode: fixed local source, moving stress test, or disabled",
    )
    parser.add_argument("--source-strength", type=float, default=0.025, help="amplitude of the source field")
    parser.add_argument("--source-sigma", type=float, default=0.65, help="Gaussian width of the source field")
    parser.add_argument("--source-period", type=float, default=6.0, help="period of the moving source stress test")
    parser.add_argument("--output-dir", type=Path, default=Path("satellite_usv_hocbf_demo"))
    parser.add_argument("--movie-format", choices=["gif", "mp4", "none"], default="gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--movie-stride", type=int, default=2, help="save one movie frame every N simulation steps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid = GridSpec(coarse_shape=(2, 2), fine_per_coarse=(4, 4))
    config = HOCBFConfig(
        steps=args.steps,
        dt=args.dt,
        target_rate=args.target_rate,
        velocity_limit=args.velocity_limit,
        control_limit=args.control_limit,
        w_limit=args.w_limit,
        source_mode=args.source_mode,
        source_strength=args.source_strength,
        source_sigma=args.source_sigma,
        source_period=args.source_period,
    )
    history = run_hocbf_simulation(config, grid)
    fields_path, metrics_path, movie_path = save_hocbf_figures(
        history,
        grid,
        args.output_dir,
        args.movie_format,
        args.fps,
        args.movie_stride,
    )

    print("satellite-USV HOCBF demo completed")
    print(f"coarse grid: {grid.coarse_shape}, fine grid: {grid.fine_shape}")
    print(f"robots: {len(history.assigned_coarse)}")
    print(f"steps: {config.steps}")
    print(f"final tracking error: {history.tracking_error[-1]:.4f}")
    print(f"final desired error: {history.desired_error[-1]:.4f}")
    print(f"minimum barrier: {min(float(barrier.min()) for barrier in history.barrier):.4e}")
    print(f"minimum psi1: {min(float(psi1.min()) for psi1 in history.psi1):.4e}")
    max_violation = max(float((-residual).clip(min=0).max()) for residual in history.hocbf_residual)
    min_relaxed = min(
        float((residual + slack).min())
        for residual, slack in zip(history.hocbf_residual, history.slack)
    )
    print(f"maximum pre-slack HOCBF violation: {max_violation:.4e}")
    print(f"maximum slack: {max(float(slack.max()) for slack in history.slack):.4e}")
    print(f"minimum relaxed HOCBF residual: {min_relaxed:.4e}")
    print(f"minimum position barrier: {min(history.position_barrier):.4e}")
    print(f"minimum intervention margin: {min(history.action_margin):.4e}")
    print(f"final path length: {history.path_length[-1]:.4f}")
    print(f"final intervention energy: {history.action_energy[-1]:.4f}")
    print(f"final velocity energy: {history.velocity_energy[-1]:.4f}")
    print(f"fields figure: {fields_path}")
    print(f"metrics figure: {metrics_path}")
    if movie_path is not None:
        print(f"movie: {movie_path}")


if __name__ == "__main__":
    main()
