import numpy as np

from satellite_usv.control import (
    ControlConfig,
    build_control_truth,
    build_upper_target,
    run_control_comparison,
    run_control_simulation,
)
from satellite_usv.model import (
    GridSpec,
    SimulationConfig,
    build_satellite_consistent_truth,
    coarse_to_fine,
    fine_to_coarse,
    run_simulation,
)


def test_averaging_matrix_matches_block_average():
    grid = GridSpec()
    fine = np.arange(grid.fine_count, dtype=np.float64).reshape(grid.fine_shape)

    h_average = grid.build_averaging_matrix() @ fine.reshape(-1)

    np.testing.assert_allclose(h_average.reshape(grid.coarse_shape), fine_to_coarse(grid, fine))


def test_default_grid_uses_25_coarse_cells():
    grid = GridSpec()

    assert grid.coarse_count == 25
    assert grid.fine_shape == (15, 15)


def test_truth_keeps_satellite_block_means():
    grid = GridSpec()
    rng = np.random.default_rng(0)
    satellite = np.arange(1, grid.coarse_count + 1, dtype=np.float64).reshape(grid.coarse_shape) / 10.0

    truth = build_satellite_consistent_truth(grid, satellite, rng)

    np.testing.assert_allclose(fine_to_coarse(grid, truth), satellite)


def test_coarse_to_fine_round_trip():
    grid = GridSpec()
    coarse = np.arange(grid.coarse_count, dtype=np.float64).reshape(grid.coarse_shape)

    fine = coarse_to_fine(grid, coarse)

    np.testing.assert_allclose(fine_to_coarse(grid, fine), coarse)


def test_run_simulation_records_expected_lengths():
    history = run_simulation(SimulationConfig(steps=3, seed=1))

    assert len(history.truth) == 4
    assert len(history.estimate) == 4
    assert len(history.uncertainty) == 4
    assert len(history.observation_density) == 4
    assert len(history.y_ref) == 4
    assert len(history.y_output) == 4
    assert len(history.usv_positions) == 4
    assert history.truth[-1].shape == GridSpec().fine_shape
    assert np.isfinite(history.fine_rmse[-1])
    assert np.isfinite(history.weighted_rmse[-1])
    assert np.isfinite(history.tracking_error[-1])
    assert history.observed_fraction[-1] > 0.0
    assert np.sum(history.control[-1]) == 0.0


def test_tracking_outputs_are_normalized():
    history = run_simulation(SimulationConfig(steps=3, seed=1, planner_mode="tracking"))

    np.testing.assert_allclose(np.sum(history.y_ref[-1]), 1.0)
    np.testing.assert_allclose(np.sum(history.y_output[-1]), 1.0)
    assert history.tracking_error[-1] >= 0.0


def test_survey_planner_keeps_moving_while_cells_are_unobserved():
    history = run_simulation(SimulationConfig(steps=20, seed=7))

    tail_positions = history.usv_positions[-5:]

    assert history.observed_fraction[-1] < 1.0
    assert any(tail_positions[index] != tail_positions[index - 1] for index in range(1, len(tail_positions)))


def test_control_demo_reduces_environment_state():
    history = run_control_simulation(ControlConfig(steps=5, seed=1, planner="hierarchical"))

    assert len(history.truth) == 6
    assert history.total_concentration[-1] < history.total_concentration[0]
    assert history.max_concentration[-1] <= history.max_concentration[0]
    assert np.isfinite(history.tracking_error[-1])


def test_control_target_keeps_upper_right_high():
    grid = GridSpec(coarse_shape=(4, 4), fine_per_coarse=(4, 4))
    initial_coarse = fine_to_coarse(grid, build_control_truth(grid))
    target = build_upper_target(grid, initial_coarse, ControlConfig())

    assert initial_coarse[0, -1] > initial_coarse[0, 0]
    assert initial_coarse[-1, -1] > initial_coarse[-1, 0]
    assert target[-1, -1] == np.max(target)
    assert target[-1, -1] > target[0, -1]
    assert target[-1, -1] > target[-1, 0]


def test_control_planner_does_not_stall_at_one_cell():
    history = run_control_simulation(ControlConfig(steps=30, seed=7, planner="hierarchical"))
    positions = [positions_at_step[0] for positions_at_step in history.usv_positions]
    stay_count = sum(
        current_position == previous_position
        for previous_position, current_position in zip(positions, positions[1:])
    )

    assert len(set(positions)) >= 10
    assert stay_count < 20


def test_multi_usv_control_planner_keeps_robots_separated():
    history = run_control_simulation(ControlConfig(steps=60, seed=7, planner="hierarchical", usv_count=3))

    for positions in history.usv_positions:
        assert len(set(positions)) == len(positions)

    for robot_index in range(3):
        path = [positions[robot_index] for positions in history.usv_positions]
        assert all(current != previous for previous, current in zip(path, path[1:]))


def test_control_comparison_uses_expected_modes():
    histories = run_control_comparison(ControlConfig(steps=2, seed=1))

    assert set(histories) == {"no_control", "random_control", "greedy_fine", "hierarchical"}
    assert histories["no_control"].control_cost[-1] == 0.0
    assert histories["hierarchical"].control_cost[-1] > 0.0
