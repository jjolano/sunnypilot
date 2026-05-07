import pytest

from openpilot.selfdrive.controls.lib.experimental_lateral_path_planner import (
  ExperimentalLateralPathPlanner,
  ExperimentalLateralPathPlannerInputs,
  ExperimentalLateralPathPlannerState,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


def make_inputs(**overrides) -> ExperimentalLateralPathPlannerInputs:
  data = {
    "enabled": True,
    "lat_active": True,
    "v_ego": 20.0,
    "baseline_curvature": 0.001,
    "measured_curvature": 0.0007,
    "previous_desired_curvature": 0.0009,
    "position_x": tuple(float(i) for i in range(ModelConstants.IDX_N)),
    "position_y": tuple(0.0 for _ in range(ModelConstants.IDX_N)),
    "position_y_std": tuple(0.1 for _ in range(ModelConstants.IDX_N)),
    "lane_line_probs": (0.0, 0.95, 0.95, 0.0),
    "left_lane_y0": -1.8,
    "right_lane_y0": 1.8,
    "left_road_edge_y0": -3.4,
    "right_road_edge_y0": 3.4,
    "lane_change_active": False,
  }
  data.update(overrides)
  return ExperimentalLateralPathPlannerInputs(**data)


def arm_planner(planner: ExperimentalLateralPathPlanner, **overrides):
  result = None
  for _ in range(4):
    result = planner.update(make_inputs(**overrides))
  assert result is not None
  assert result.state == ExperimentalLateralPathPlannerState.active
  return result


def test_disabled_returns_baseline_and_resets_state():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  arm_planner(planner)

  result = planner.update(make_inputs(enabled=False, baseline_curvature=0.002))

  assert result.state == ExperimentalLateralPathPlannerState.baseline
  assert not result.active
  assert result.reason == "disabled"
  assert result.desired_curvature == pytest.approx(0.002)


def test_requires_stable_valid_window_before_applying_candidate():
  planner = ExperimentalLateralPathPlanner(dt=0.1)

  first = planner.update(make_inputs(position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N))))
  second = planner.update(make_inputs(position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N))))
  third = planner.update(make_inputs(position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N))))
  fourth = planner.update(make_inputs(position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N))))

  assert first.state == ExperimentalLateralPathPlannerState.arming
  assert first.desired_curvature == pytest.approx(0.001)
  assert second.desired_curvature == pytest.approx(0.001)
  assert third.desired_curvature == pytest.approx(0.001)
  assert fourth.state == ExperimentalLateralPathPlannerState.active
  assert fourth.active
  assert fourth.desired_curvature != pytest.approx(0.001)


def test_short_invalid_burst_holds_last_experimental_output():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  active = arm_planner(planner, position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N)))

  result = planner.update(make_inputs(position_x=(0.0, 1.0), position_y=(0.0, 0.1), baseline_curvature=-0.002))

  assert result.state == ExperimentalLateralPathPlannerState.degraded_hold
  assert result.active
  assert result.reason == "invalid_path_hold"
  assert result.desired_curvature == pytest.approx(active.desired_curvature)


def test_persistent_invalidity_blends_back_to_baseline_before_cooldown():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  active = arm_planner(planner, position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N)))
  invalid_inputs = make_inputs(position_x=(0.0, 1.0), position_y=(0.0, 0.1), baseline_curvature=-0.002)

  for _ in range(4):
    result = planner.update(invalid_inputs)

  assert result.state == ExperimentalLateralPathPlannerState.blending_to_baseline
  assert result.reason == "invalid_path_blending"
  assert -0.002 < result.desired_curvature < active.desired_curvature


def test_cooldown_prevents_immediate_reentry_after_fallback():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  arm_planner(planner, position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N)))
  invalid_inputs = make_inputs(position_x=(0.0, 1.0), position_y=(0.0, 0.1), baseline_curvature=-0.002)

  for _ in range(12):
    result = planner.update(invalid_inputs)
  assert result.state == ExperimentalLateralPathPlannerState.cooldown

  valid_result = planner.update(make_inputs(position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N)), baseline_curvature=-0.002))

  assert valid_result.state == ExperimentalLateralPathPlannerState.cooldown
  assert valid_result.desired_curvature == pytest.approx(-0.002)


def test_low_speed_resets_to_baseline():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  arm_planner(planner)

  result = planner.update(make_inputs(v_ego=1.0, baseline_curvature=0.004))

  assert result.state == ExperimentalLateralPathPlannerState.baseline
  assert result.reason == "low_speed"
  assert result.desired_curvature == pytest.approx(0.004)


def test_high_lane_uncertainty_returns_baseline_without_arming():
  planner = ExperimentalLateralPathPlanner(dt=0.1)

  result = planner.update(make_inputs(position_y_std=tuple(1.2 for _ in range(ModelConstants.IDX_N))))

  assert result.state == ExperimentalLateralPathPlannerState.baseline
  assert result.reason == "low_confidence"
  assert result.desired_curvature == pytest.approx(0.001)


def test_nonfinite_path_uncertainty_returns_baseline_without_arming():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  position_y_std = [0.1 for _ in range(ModelConstants.IDX_N)]
  position_y_std[4] = float("nan")

  result = planner.update(make_inputs(position_y_std=tuple(position_y_std)))

  assert result.state == ExperimentalLateralPathPlannerState.baseline
  assert result.reason == "low_confidence"
  assert result.desired_curvature == pytest.approx(0.001)


def test_malformed_path_uncertainty_returns_baseline_without_arming():
  planner = ExperimentalLateralPathPlanner(dt=0.1)

  result = planner.update(make_inputs(position_y_std=(0.1, 0.1)))

  assert result.state == ExperimentalLateralPathPlannerState.baseline
  assert result.reason == "low_confidence"
  assert result.desired_curvature == pytest.approx(0.001)


def test_nonfinite_lane_probability_suppresses_lane_center_bias():
  planner = ExperimentalLateralPathPlanner(dt=0.1)

  result = arm_planner(
    planner,
    baseline_curvature=0.0,
    position_y=tuple(0.4 for _ in range(ModelConstants.IDX_N)),
    lane_line_probs=(0.0, float("nan"), 0.95, 0.0),
    left_road_edge_y0=None,
    right_road_edge_y0=None,
  )

  assert result.active
  assert result.desired_curvature == pytest.approx(0.0)


def test_lane_center_bias_is_capped_to_subtle_lateral_accel():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  v_ego = 17.0

  result = arm_planner(
    planner,
    v_ego=v_ego,
    baseline_curvature=0.0,
    measured_curvature=0.0,
    previous_desired_curvature=0.0,
    position_y=tuple(0.42 for _ in range(ModelConstants.IDX_N)),
    left_road_edge_y0=None,
    right_road_edge_y0=None,
  )

  assert result.active
  assert result.reason == "ok"
  assert abs(result.desired_curvature) * v_ego ** 2 <= 0.25 + 1e-9


def test_near_road_edge_biases_candidate_away_from_edge_after_arming():
  planner = ExperimentalLateralPathPlanner(dt=0.1)

  result = arm_planner(
    planner,
    baseline_curvature=0.0,
    position_y=tuple(1.3 for _ in range(ModelConstants.IDX_N)),
    left_road_edge_y0=-3.4,
    right_road_edge_y0=1.7,
  )

  assert result.active
  assert result.reason == "ok"
  assert result.desired_curvature < 0.0


def test_road_edge_bias_is_capped_to_subtle_lateral_accel():
  planner = ExperimentalLateralPathPlanner(dt=0.1)
  v_ego = 20.0

  result = arm_planner(
    planner,
    v_ego=v_ego,
    baseline_curvature=0.0,
    measured_curvature=0.0,
    previous_desired_curvature=0.0,
    position_y=tuple(1.3 for _ in range(ModelConstants.IDX_N)),
    lane_line_probs=(0.0, 0.0, 0.0, 0.0),
    left_road_edge_y0=-3.4,
    right_road_edge_y0=1.7,
  )

  assert result.active
  assert result.reason == "ok"
  assert abs(result.desired_curvature) * v_ego ** 2 <= 0.35 + 1e-9
