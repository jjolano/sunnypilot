import pytest

from cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.lane_change_path_shaper import ENTRY_BLEND_DURATION, EXIT_BLEND_DURATION, LANE_CHANGE_DURATION, LaneChangePathShaper, LaneChangePathShaperInputs


LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection


def make_inputs(
  lat_active: bool = True,
  v_ego: float = 30.0,
  left_blinker: bool = True,
  right_blinker: bool = False,
  steering_pressed: bool = False,
  lane_change_state: int = LaneChangeState.laneChangeStarting,
  lane_change_direction: int = LaneChangeDirection.left,
  model_curvature: float = 0.0,
  prev_desired_curvature: float = 0.0,
  lane_line_probs: tuple[float, float, float, float] = (0.0, 0.95, 0.96, 0.0),
  left_lane_y0: float | None = -1.8,
  right_lane_y0: float | None = 1.8,
) -> LaneChangePathShaperInputs:
  return LaneChangePathShaperInputs(
    lat_active=lat_active,
    v_ego=v_ego,
    left_blinker=left_blinker,
    right_blinker=right_blinker,
    steering_pressed=steering_pressed,
    lane_change_state=lane_change_state,
    lane_change_direction=lane_change_direction,
    model_curvature=model_curvature,
    prev_desired_curvature=prev_desired_curvature,
    lane_line_probs=lane_line_probs,
    left_lane_y0=left_lane_y0,
    right_lane_y0=right_lane_y0,
  )


def run_steps(controller: LaneChangePathShaper, inputs: LaneChangePathShaperInputs, duration: float):
  steps = int(duration / DT_CTRL)
  return [controller.update(inputs) for _ in range(steps)]


def run_speed_profile(controller: LaneChangePathShaper, speeds: list[float]):
  return [controller.update(make_inputs(v_ego=speed)) for speed in speeds]


def test_lane_change_profile_is_symmetric():
  left_controller = LaneChangePathShaper()
  right_controller = LaneChangePathShaper()

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
  controller = LaneChangePathShaper()
  inputs = make_inputs()

  results = run_steps(controller, inputs, LANE_CHANGE_DURATION + EXIT_BLEND_DURATION + 0.5)

  assert results[0].desired_curvature == pytest.approx(inputs.model_curvature)
  assert results[-1].blend == pytest.approx(0.0)
  assert results[-1].desired_curvature == pytest.approx(inputs.model_curvature)


def test_peak_highway_curvature_is_smoother():
  controller = LaneChangePathShaper()
  inputs = make_inputs()

  results = run_steps(controller, inputs, LANE_CHANGE_DURATION)
  peak_curvature = max(abs(result.desired_curvature - inputs.model_curvature) for result in results)

  assert peak_curvature < 0.001


def test_decel_does_not_amplify_lane_change_curvature():
  steps = int(LANE_CHANGE_DURATION / DT_CTRL)
  constant_results = run_speed_profile(LaneChangePathShaper(), [30.0] * steps)
  decel_speeds = [30.0 - 10.0 * i / max(steps - 1, 1) for i in range(steps)]
  decel_results = run_speed_profile(LaneChangePathShaper(), decel_speeds)

  constant_peak = max(abs(result.desired_curvature) for result in constant_results)
  decel_peak = max(abs(result.desired_curvature) for result in decel_results)

  assert decel_peak <= constant_peak * 1.05


def test_same_direction_model_curvature_remains_primary():
  controller = LaneChangePathShaper()
  inputs = make_inputs(model_curvature=-0.0012)

  result = run_steps(controller, inputs, 1.0)[-1]

  assert result.active
  assert result.desired_curvature == pytest.approx(inputs.model_curvature)


def test_initial_turn_in_authority_is_limited_and_symmetric():
  left_controller = LaneChangePathShaper()
  right_controller = LaneChangePathShaper()

  left_inputs = make_inputs()
  right_inputs = make_inputs(left_blinker=False, right_blinker=True, lane_change_direction=LaneChangeDirection.right)

  left_result = run_steps(left_controller, left_inputs, 1.0)[-1]
  right_result = run_steps(right_controller, right_inputs, 1.0)[-1]

  left_delta = left_result.desired_curvature - left_inputs.model_curvature
  right_delta = right_result.desired_curvature - right_inputs.model_curvature
  assert left_delta < 0.0
  assert right_delta > 0.0
  assert abs(left_delta) < 0.0003
  assert abs(right_delta) < 0.0003
  assert left_delta == pytest.approx(-right_delta, rel=1e-6)


def test_lane_width_geometry_is_damped_toward_nominal():
  narrow_controller = LaneChangePathShaper()
  wide_controller = LaneChangePathShaper()

  narrow_inputs = make_inputs(left_lane_y0=-1.5, right_lane_y0=1.5)
  wide_inputs = make_inputs(left_lane_y0=-2.0, right_lane_y0=2.0)

  narrow_result = run_steps(narrow_controller, narrow_inputs, 1.0)[-1]
  wide_result = run_steps(wide_controller, wide_inputs, 1.0)[-1]

  narrow_delta = abs(narrow_result.desired_curvature - narrow_inputs.model_curvature)
  wide_delta = abs(wide_result.desired_curvature - wide_inputs.model_curvature)

  assert wide_delta > narrow_delta
  assert wide_delta / narrow_delta < 1.25


def test_soft_fallback_blends_back_to_model_curvature():
  controller = LaneChangePathShaper()
  inputs = make_inputs()

  engaged = run_steps(controller, inputs, ENTRY_BLEND_DURATION + 0.1)
  assert engaged[-1].blend > 0.8

  fallback_inputs = make_inputs(lane_line_probs=(0.0, 0.2, 0.2, 0.0))
  fallback_results = run_steps(controller, fallback_inputs, EXIT_BLEND_DURATION + 0.2)

  assert fallback_results[0].soft_fallback
  assert 0.0 < fallback_results[0].blend < engaged[-1].blend
  assert fallback_results[-1].blend == pytest.approx(0.0)
  assert fallback_results[-1].desired_curvature == pytest.approx(fallback_inputs.model_curvature)


def test_early_finishing_blends_back_to_model_curvature():
  controller = LaneChangePathShaper()
  inputs = make_inputs()

  engaged = run_steps(controller, inputs, ENTRY_BLEND_DURATION + 0.1)
  assert engaged[-1].blend > 0.8

  finishing_inputs = make_inputs(lane_change_state=LaneChangeState.laneChangeFinishing, model_curvature=0.0003)
  finishing_results = run_steps(controller, finishing_inputs, EXIT_BLEND_DURATION + 0.2)

  assert finishing_results[0].blend < engaged[-1].blend
  assert finishing_results[-1].blend == pytest.approx(0.0)
  assert finishing_results[-1].desired_curvature == pytest.approx(finishing_inputs.model_curvature)


@pytest.mark.parametrize("abort_overrides", [
  {"lat_active": False},
  {"left_blinker": False, "right_blinker": False},
  {"left_blinker": True, "right_blinker": True},
  {"left_blinker": False, "right_blinker": True, "lane_change_direction": LaneChangeDirection.left},
])
def test_hard_abort_resets_immediately(abort_overrides):
  controller = LaneChangePathShaper()
  inputs = make_inputs()
  run_steps(controller, inputs, 1.0)

  abort_inputs = make_inputs(model_curvature=0.0003, **abort_overrides)
  result = controller.update(abort_inputs)

  assert result.blend == pytest.approx(0.0)
  assert not result.active
  assert result.desired_curvature == pytest.approx(abort_inputs.model_curvature)


def test_manual_torque_start_does_not_abort_path_shaping():
  controller = LaneChangePathShaper()

  result = controller.update(make_inputs(steering_pressed=True))

  assert result.blend > 0.0
  assert result.active


def test_sustained_manual_torque_soft_fallbacks_path_shaping():
  controller = LaneChangePathShaper()
  inputs = make_inputs()

  engaged = run_steps(controller, inputs, ENTRY_BLEND_DURATION + 0.1)
  assert engaged[-1].blend > 0.8

  override_inputs = make_inputs(steering_pressed=True, model_curvature=0.0003)
  fallback_results = run_steps(controller, override_inputs, EXIT_BLEND_DURATION + 0.2)

  assert fallback_results[0].soft_fallback
  assert 0.0 < fallback_results[0].blend < engaged[-1].blend
  assert fallback_results[-1].blend == pytest.approx(0.0)
  assert fallback_results[-1].desired_curvature == pytest.approx(override_inputs.model_curvature)


def test_ineligible_entry_stays_model_driven():
  controller = LaneChangePathShaper()
  result = controller.update(make_inputs(prev_desired_curvature=0.002))
  assert result.blend == pytest.approx(0.0)
  assert result.desired_curvature == pytest.approx(0.0)

  follow_up = controller.update(make_inputs())
  assert follow_up.blend == pytest.approx(0.0)
  assert follow_up.desired_curvature == pytest.approx(0.0)
