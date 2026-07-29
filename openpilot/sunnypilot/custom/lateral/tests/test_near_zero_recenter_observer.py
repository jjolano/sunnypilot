from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.lateral.near_zero_recenter_observer import NearZeroRecenterObserver


DT = 0.01


def base(**overrides):
  values = dict(
    active=True,
    v_ego=20.0,
    steering_pressed=False,
    steer_limited_by_safety=False,
    curvature_limited=False,
    desired_lateral_accel=-0.04,
    actual_lateral_accel=0.10,
    steering_rate_deg=0.0,
    output_torque=-0.2,
    steer_max=1.0,
  )
  values.update(overrides)
  return values


def test_detects_near_zero_recenter_conflict_and_duration():
  obs = NearZeroRecenterObserver(DT)

  first = obs.update(**base())
  second = obs.update(**base(actual_lateral_accel=0.09))

  assert first.conflict is True
  assert first.error == pytest.approx(-0.14)
  assert first.closingRate == 0.0
  assert first.duration == pytest.approx(DT)
  assert second.conflict is True
  assert second.closingRate > 0.0
  assert second.duration == pytest.approx(2.0 * DT)


@pytest.mark.parametrize("override", [
  {"active": False},
  {"v_ego": 9.9},
  {"steering_pressed": True},
  {"steer_limited_by_safety": True},
  {"curvature_limited": True},
  {"desired_lateral_accel": 0.08},
  {"actual_lateral_accel": 0.05},
  {"actual_lateral_accel": 0.16},
  {"desired_lateral_accel": 0.04},
  {"steering_rate_deg": 20.1},
  {"output_torque": -0.5},
  {"output_torque": 0.2},
  {"steer_max": 0.0},
])
def test_gates_fail_closed(override):
  obs = NearZeroRecenterObserver(DT)

  debug = obs.update(**base(**override))

  assert debug.conflict is False
  assert debug.duration == 0.0


def test_nonfinite_inputs_fail_closed_and_reset():
  obs = NearZeroRecenterObserver(DT)
  assert obs.update(**base()).conflict is True

  debug = obs.update(**base(steering_rate_deg=math.nan))
  next_debug = obs.update(**base())

  assert debug.conflict is False
  assert next_debug.conflict is True
  assert next_debug.closingRate == 0.0
