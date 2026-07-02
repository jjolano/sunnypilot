import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.drive_lab.roll_comp_profile import (
  GRAVITY,
  build_roll_comp_profile,
  load_roll_comp_profile,
  render_roll_comp_profile,
  save_roll_comp_profile,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


class FakeUnion(SimpleNamespace):
  def which(self):
    return "torqueState"


def _msg(kind: str, t_s: float, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def _frame(t_s: float, *, roll: float = 0.0, p: float = 0.0, i: float = 0.0, f: float = 0.0,
           desired_lateral_accel: float = 0.0, lat_active: bool = True, steering_pressed: bool = False,
           v_ego: float = 20.0, saturated: bool = False):
  torque_state = SimpleNamespace(
    active=lat_active,
    p=p,
    i=i,
    f=f,
    desiredLateralAccel=desired_lateral_accel,
    saturated=saturated,
  )
  return [
    _msg("carState", t_s, vEgo=v_ego, steeringPressed=steering_pressed),
    _msg("carControl", t_s, latActive=lat_active),
    _msg("liveParameters", t_s, roll=roll),
    _msg("controlsState", t_s, lateralControlState=FakeUnion(torqueState=torque_state)),
  ]


def _roll_to_x(roll: float) -> float:
  return -math.sin(roll) * GRAVITY


def test_roll_comp_profile_recovers_known_slope():
  true_slope = 0.55
  offset = -0.19
  rolls = np.linspace(-0.08, 0.08, 100)
  msgs = []
  for i, roll in enumerate(rolls):
    x = _roll_to_x(roll)
    p = true_slope * x
    i_val = offset
    msgs.extend(_frame(i * 0.05, roll=roll, p=p, i=i_val, f=0.0))

  report = build_roll_comp_profile(msgs, source="synthetic")
  rendered = render_roll_comp_profile(report)

  assert report.point_count == len(rolls)
  assert report.slope == pytest.approx(true_slope, rel=1e-3)
  assert report.integrator_mean == pytest.approx(offset, abs=1e-6)
  assert report.roll_span > 0.5
  assert "Roll compensation profile for synthetic" in rendered


def test_roll_comp_profile_gates_exclude_bad_frames():
  good_rolls = np.linspace(-0.06, 0.06, 40)
  bad_frames = [
    dict(lat_active=False),
    dict(steering_pressed=True),
    dict(v_ego=10.0),
    dict(desired_lateral_accel=0.5),
    dict(saturated=True),
  ]
  msgs = []
  t = 0.0
  for roll in good_rolls:
    x = _roll_to_x(roll)
    msgs.extend(_frame(t, roll=roll, p=0.5 * x, i=-0.1))
    t += 0.05

  for bad in bad_frames:
    msgs.extend(_frame(t, roll=0.0, p=10.0, i=10.0, **bad))
    t += 0.05

  report = build_roll_comp_profile(msgs, source="gated")

  assert report.point_count == len(good_rolls)
  assert report.slope == pytest.approx(0.5, rel=1e-2)
  assert report.integrator_mean == pytest.approx(-0.1, abs=1e-5)


def test_roll_comp_profile_delta_gate_excludes_large_transitions():
  msgs = [
    *_frame(0.0, roll=0.0, p=0.0, i=0.0, desired_lateral_accel=0.0),
    *_frame(0.05, roll=0.02, p=10.0, i=10.0, desired_lateral_accel=0.10),
    *_frame(0.10, roll=0.03, p=0.0, i=0.0, desired_lateral_accel=0.10),
  ]
  report = build_roll_comp_profile(msgs, source="delta-gate")
  assert report.point_count == 2


def test_roll_comp_profile_degenerate_span_no_slope():
  msgs = []
  for i in range(20):
    msgs.extend(_frame(i * 0.05, roll=0.0, p=0.2, i=-0.1))

  report = build_roll_comp_profile(msgs, source="degenerate")

  assert report.point_count == 20
  assert report.slope is None
  assert report.roll_span == pytest.approx(0.0, abs=1e-9)


def test_roll_comp_profile_save_and_load_round_trip(tmp_path):
  msgs = []
  for i, roll in enumerate(np.linspace(-0.05, 0.05, 30)):
    x = _roll_to_x(roll)
    msgs.extend(_frame(i * 0.05, roll=roll, p=0.6 * x, i=-0.05))

  report = build_roll_comp_profile(msgs, source="roundtrip")
  path = tmp_path / "roll-comp-profile.json"
  save_roll_comp_profile(report, path)
  loaded = load_roll_comp_profile(path)

  assert loaded == report
