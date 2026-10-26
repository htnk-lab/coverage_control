#!/usr/bin/env python

import argparse
from pathlib import Path

from .cbf_control import CBFConfig, run_cbf_simulation
from .cbf_plotting import save_cbf_figures
from .model import GridSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the first-order CBF toy demo.")
    parser.add_argument("--steps", type=int, default=80, help="number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.15, help="time step")
    parser.add_argument("--control-limit", type=float, default=2.0, help="upper bound for each coarse-cell input")
    parser.add_argument("--cbf-gamma", type=float, default=6.0, help="class-K gain in dot b + gamma b >= 0")
    parser.add_argument("--cbf-epsilon", type=float, default=0.08, help="tracking tube radius")
    parser.add_argument("--slack-penalty", type=float, default=1.0e6, help="quadratic penalty for CBF slack")
    parser.add_argument("--target-rate", type=float, default=0.55, help="decay rate of the upper reference")
    parser.add_argument("--output-dir", type=Path, default=Path("satellite_usv_cbf_demo"))
    parser.add_argument("--movie-format", choices=["gif", "mp4", "none"], default="gif")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--movie-stride", type=int, default=1, help="save one movie frame every N simulation steps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid = GridSpec(coarse_shape=(2, 2), fine_per_coarse=(4, 4))
    config = CBFConfig(
        steps=args.steps,
        dt=args.dt,
        control_limit=args.control_limit,
        cbf_gamma=args.cbf_gamma,
        cbf_epsilon=args.cbf_epsilon,
        slack_penalty=args.slack_penalty,
        target_rate=args.target_rate,
    )
    history = run_cbf_simulation(config, grid)
    fields_path, metrics_path, movie_path = save_cbf_figures(
        history,
        grid,
        args.output_dir,
        args.movie_format,
        args.fps,
        args.movie_stride,
    )

    print("satellite-USV first-order CBF demo completed")
    print(f"coarse grid: {grid.coarse_shape}, fine grid: {grid.fine_shape}")
    print(f"robots: {len(history.assigned_coarse)}")
    print(f"steps: {config.steps}")
    print(f"final tracking error: {history.tracking_error[-1]:.4f}")
    print(f"final desired error: {history.desired_error[-1]:.4f}")
    print(f"minimum final barrier: {history.barrier[-1].min():.4f}")
    print(f"maximum pre-slack continuous CBF violation: {max(history.cbf_violation):.4e}")
    print(f"maximum slack: {max(float(slack.max()) for slack in history.slack):.4e}")
    print(
        "minimum relaxed continuous CBF residual: "
        f"{min(float((margin + slack).min()) for margin, slack in zip(history.cbf_margin, history.slack)):.4e}"
    )
    print(
        "minimum Euler discrete CBF residual: "
        f"{min(float(residual.min()) for residual in history.discrete_cbf_residual):.4e}"
    )
    print(f"fields figure: {fields_path}")
    print(f"metrics figure: {metrics_path}")
    if movie_path is not None:
        print(f"movie: {movie_path}")


if __name__ == "__main__":
    main()
