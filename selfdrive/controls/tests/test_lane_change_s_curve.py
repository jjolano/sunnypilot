import pytest

from cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.lane_change_s_curve import EXIT_BLEND_DURATION, LANE_CHANGE_DURATION, LaneChangeSCurveController, LaneChangeSCurveInputs


LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection


def make_inputs(**overrides) -> LaneChangeSCurveInputs:
  data = {
    "lat_active": True,
    "v_ego": 30.0,
    "left_blinker": True,
    "right_blinker": False,
    "steering_pressed": False,
    "lane_change_state": LaneChangeState.laneChangeStarting,
    "lane_change_direction": LaneChangeDirection.left,
    "model_curvature": 0.0,
    "prev_desired_curvature": 0.0,
    "lane_line_probs": (0.0, 0.95, 0.96, 0.0),
    "left_lane_y0": -1.8,
    "right_lane_y0": 1.8,
  }
  data.update(overrides)
  return LaneChangeSCurveInputs(**data)


def run_steps(controller: LaneChangeSCurveController, inputs: LaneChangeSCurveInputs, duration: float):
  steps = int(duration / DT_CTRL)
  return [controller.update(inputs) for _ in range(steps)]


def test_lane_change_profile_is_symmetric():
  left_controller = LaneChangeSCurveController()
  right_controller = LaneChangeSCurveController()

  left_inputs = make_inputs()
  right_inputs = make_inputs(left_blinker=False, right_blinker=True, lane_change_direction=LaneChangeDirection.right)

  left_result = run_steps(left_controller, left_inputs, 1.0)[-1]
  right_result = run_steps(right_controller, right_inputs, 1.0)[-1]

  left_delta = left_result.desired_curvature - left_inputs.model_curvature
  right_delta = right_result.desired_curvature - right_inputs.model_curvature
  assert left_delta < 0.0
  assert right_delta > 0.0
  assert left_delta == pytest.approx(-right_delta, rel=1e-6)


def test_lane_change_blends_back_after_profile_completion():
  controller = LaneChangeSCurveController()
  inputs = make_inputs()

  results = run_steps(controller, inputs, LANE_CHANGE_DURATION + EXIT_BLEND_DURATION + 0.5)

  assert results[0].desired_curvature == pytest.approx(inputs.model_curvature)
  assert results[-1].blend == pytest.approx(0.0)
  assert results[-1].desired_curvature == pytest.approx(inputs.model_curvature)


def test_peak_highway_curvature_is_smoother():
  controller = LaneChangeSCurveController()
  inputs = make_inputs()

  results = run_steps(controller, inputs, LANE_CHANGE_DURATION)
  peak_curvature = max(abs(result.desired_curvature - inputs.model_curvature) for result in results)

  assert peak_curvature < 0.001


def test_soft_fallback_blends_back_to_model_curvature():
  controller = LaneChangeSCurveController()
  inputs = make_inputs()

  engaged = run_steps(controller, inputs, 1.0)
  assert engaged[-1].blend > 0.8

  fallback_inputs = make_inputs(lane_line_probs=(0.0, 0.2, 0.2, 0.0))
  fallback_results = run_steps(controller, fallback_inputs, EXIT_BLEND_DURATION + 0.2)

  assert fallback_results[0].soft_fallback
  assert 0.0 < fallback_results[0].blend < engaged[-1].blend
  assert fallback_results[-1].blend == pytest.approx(0.0)
  assert fallback_results[-1].desired_curvature == pytest.approx(fallback_inputs.model_curvature)


def test_hard_abort_resets_immediately():
  controller = LaneChangeSCurveController()
  inputs = make_inputs()
  run_steps(controller, inputs, 1.0)

  result = controller.update(make_inputs(steering_pressed=True))

  assert result.blend == pytest.approx(0.0)
  assert not result.active
  assert result.desired_curvature == pytest.approx(inputs.model_curvature)


def test_ineligible_entry_stays_model_driven():
  controller = LaneChangeSCurveController()
  result = controller.update(make_inputs(prev_desired_curvature=0.002))
  assert result.blend == pytest.approx(0.0)
  assert result.desired_curvature == pytest.approx(0.0)

  follow_up = controller.update(make_inputs())
  assert follow_up.blend == pytest.approx(0.0)
  assert follow_up.desired_curvature == pytest.approx(0.0)
