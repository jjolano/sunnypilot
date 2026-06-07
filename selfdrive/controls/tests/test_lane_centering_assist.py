import pytest

from openpilot.selfdrive.controls.lib.lane_centering_assist import (
  LANE_CENTERING_ASSIST_MAX_LAT_ACCEL,
  LaneCenteringAssistInputs,
  LaneCenteringAssistTracker,
)
from openpilot.selfdrive.controls.lib.lateral_demand import DEMAND_SOURCE_LATERAL_MANEUVER, DEMAND_SOURCE_MODEL_PATH


def make_inputs(**overrides):
  xs = tuple(float(i) for i in range(40))
  values = {
    "lat_active": True,
    "v_ego": 20.0,
    "measured_curvature": 0.0,
    "model_curvature": 0.0,
    "previous_processed_curvature": 0.0,
    "path_quality": 1.0,
    "path_reason": "ok",
    "lane_change_shaping_active": False,
    "lane_change_blend": 0.0,
    "curvature_limited": False,
    "steering_pressed": False,
    "left_blinker": False,
    "right_blinker": False,
    "position_x": xs,
    "position_y": tuple(0.02 + 0.006 * x for x in xs),
    "orientation_z": tuple(0.002 for _ in xs),
    "lane_line_probs": (0.0, 0.9, 0.9, 0.0),
    "demand_source": DEMAND_SOURCE_MODEL_PATH,
  }
  values.update(overrides)
  return LaneCenteringAssistInputs(**values)


def test_small_growing_error_activates_early():
  result = LaneCenteringAssistTracker().update(make_inputs(), dt=0.5)

  assert result.active
  assert result.reason == "growing_lateral_error"
  assert result.curvature_nudge > 0.0
  assert abs(result.curvature_nudge) <= LANE_CENTERING_ASSIST_MAX_LAT_ACCEL / 20.0**2
  assert result.predicted_lateral_error > result.lateral_error


def test_small_shrinking_error_releases_existing_nudge():
  tracker = LaneCenteringAssistTracker()
  first = tracker.update(make_inputs(), dt=0.5)

  shrinking = tracker.update(
    make_inputs(position_y=tuple(0.18 - 0.004 * x for x in make_inputs().position_x), orientation_z=tuple(-0.002 for _ in make_inputs().position_x)),
    dt=0.05,
  )

  assert first.active
  assert shrinking.active
  assert shrinking.reason == "error_not_growing"
  assert abs(shrinking.curvature_nudge) < abs(first.curvature_nudge)


def test_deadband_prevents_twitch_on_tiny_noisy_errors():
  tracker = LaneCenteringAssistTracker()
  signs = []

  for sign in (1, -1, 1, -1):
    result = tracker.update(
      make_inputs(position_y=tuple(sign * 0.005 for _ in make_inputs().position_x), orientation_z=tuple(0.0 for _ in make_inputs().position_x)),
      dt=0.2,
    )
    signs.append(1 if result.curvature_nudge > 0.0 else (-1 if result.curvature_nudge < 0.0 else 0))

  assert signs == [0, 0, 0, 0]


def test_hysteresis_releases_before_rapid_sign_flip():
  tracker = LaneCenteringAssistTracker()
  first = tracker.update(make_inputs(), dt=0.5)
  crossing = tracker.update(
    make_inputs(position_y=tuple(-0.02 - 0.006 * x for x in make_inputs().position_x), orientation_z=tuple(-0.002 for _ in make_inputs().position_x)),
    dt=0.1,
  )

  assert first.curvature_nudge > 0.0
  assert crossing.active
  assert crossing.curvature_nudge >= 0.0
  assert crossing.reason == "sign_hysteresis"


@pytest.mark.parametrize(
  "overrides, reason",
  [
    ({"path_quality": 0.4}, "low_path_quality"),
    ({"path_reason": "path_disagreement"}, "path_reason"),
    ({"lane_change_shaping_active": True}, "lane_change"),
    ({"lane_change_blend": 0.2}, "lane_change"),
    ({"lane_change_blend": float("nan")}, "lane_change"),
    ({"demand_source": DEMAND_SOURCE_LATERAL_MANEUVER}, "non_model_demand"),
    ({"curvature_limited": True}, "curvature_limited"),
    ({"steering_pressed": True}, "driver_steering"),
  ],
)
def test_gates_block_nudge(overrides, reason):
  result = LaneCenteringAssistTracker().update(make_inputs(**overrides), dt=0.5)

  assert not result.active
  assert result.curvature_nudge == pytest.approx(0.0)
  assert result.reason == reason


def test_hard_gate_resets_existing_nudge():
  tracker = LaneCenteringAssistTracker()
  active = tracker.update(make_inputs(), dt=0.5)
  blocked = tracker.update(make_inputs(steering_pressed=True), dt=0.05)
  released = tracker.update(
    make_inputs(position_y=tuple(0.18 - 0.004 * x for x in make_inputs().position_x), orientation_z=tuple(-0.002 for _ in make_inputs().position_x)),
    dt=0.05,
  )

  assert active.active
  assert not blocked.active
  assert blocked.curvature_nudge == pytest.approx(0.0)
  assert blocked.reason == "driver_steering"
  assert not released.active
  assert released.curvature_nudge == pytest.approx(0.0)


def test_high_speed_has_smaller_curvature_nudge_for_same_lateral_error():
  xs = make_inputs().position_x
  path_y = tuple(0.20 + 0.002 * x for x in xs)
  low_speed = LaneCenteringAssistTracker().update(make_inputs(v_ego=10.0, position_y=path_y), dt=5.0)
  high_speed = LaneCenteringAssistTracker().update(make_inputs(v_ego=30.0, position_y=path_y), dt=5.0)

  assert low_speed.active
  assert high_speed.active
  assert abs(high_speed.curvature_nudge) < abs(low_speed.curvature_nudge)
  assert abs(high_speed.curvature_nudge) <= LANE_CENTERING_ASSIST_MAX_LAT_ACCEL / 30.0**2
