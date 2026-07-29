from types import SimpleNamespace

import pytest

from openpilot.tools.drive_lab import lateral_oscillation_profile
from openpilot.tools.drive_lab.lateral_oscillation_profile import (
  build_lateral_oscillation_profile,
  load_lateral_profile,
  render_lateral_profile,
  save_lateral_profile,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


class FakeUnion(SimpleNamespace):
  def which(self):
    return "torqueState"


class FakeEnum:
  def __init__(self, name: str):
    self.name = name


def msg(kind, t_s, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def controls_msg(t_s, *, raw_curvature: float, steering_output: float = 0.0, active: bool = True):
  torque_state = SimpleNamespace(active=active, output=steering_output)
  return msg(
    "controlsState",
    t_s,
    curvature=raw_curvature * 0.95,
    desiredCurvature=raw_curvature,
    lateralControlState=FakeUnion(torqueState=torque_state),
    modelPathState=SimpleNamespace(
      rawDesiredCurvature=raw_curvature,
      processedDesiredCurvature=raw_curvature,
      gated=False,
      quality=1.0,
    ),
  )


def test_build_lateral_profile_ranks_straight_oscillation_windows(tmp_path):
  msgs = []
  for i in range(80):
    t = i * 0.5
    raw = 0.0012 if i % 8 < 4 else -0.0012
    steering_angle = -raw * 3000.0
    requested_torque = steering_angle / 20.0
    msgs.extend([
      msg("carState", t, vEgo=15.0, steeringPressed=False, leftBlinker=False, rightBlinker=False,
          steeringAngleDeg=steering_angle, steeringTorque=0.1 * steering_angle, steeringTorqueEps=100.0 * requested_torque),
      msg("carControl", t, latActive=True, actuators=SimpleNamespace(torque=requested_torque)),
      msg("carOutput", t, actuatorsOutput=SimpleNamespace(torque=requested_torque)),
      msg("modelV2", t, meta=SimpleNamespace(laneChangeState=FakeEnum("off"))),
      controls_msg(t, raw_curvature=raw, steering_output=requested_torque),
    ])

  profile = build_lateral_oscillation_profile(msgs, source="route", window_s=20.0, step_s=10.0)
  path = tmp_path / "lateral-profile.json"
  save_lateral_profile(profile, path)
  loaded = load_lateral_profile(path)
  rendered = render_lateral_profile(loaded)

  assert loaded.source == "route"
  assert loaded.straight_sample_count > 0
  assert loaded.straight_steering_angle_pp > 6.0
  assert loaded.raw_actual_corr == pytest.approx(1.0)
  assert loaded.raw_steering_corr == pytest.approx(-1.0)
  assert loaded.straight_requested_torque_pp > 0.2
  assert loaded.straight_eps_torque_pp > 20.0
  assert loaded.straight_command_eps_corr == pytest.approx(1.0)
  assert loaded.top_windows
  assert loaded.top_windows[0].eps_torque_pp > 20.0
  assert "Top windows" in rendered
  assert "eps torque pp" in rendered


def test_lateral_profile_accepts_unknown_lane_state_when_blinker_off():
  msgs = []
  for i in range(8):
    t = float(i)
    raw = 0.0004 if i % 2 == 0 else -0.0004
    msgs.extend([
      msg("carState", t, vEgo=16.0, steeringPressed=False, leftBlinker=False, rightBlinker=False,
          steeringAngleDeg=raw * 1000.0, steeringTorque=0.0, steeringTorqueEps=raw * 10000.0),
      msg("carControl", t, latActive=True, actuators=SimpleNamespace(torque=raw * 100.0)),
      msg("carOutput", t, actuatorsOutput=SimpleNamespace(torque=raw * 100.0)),
      controls_msg(t, raw_curvature=raw),
    ])

  profile = build_lateral_oscillation_profile(msgs, source="qlog-like", window_s=4.0, step_s=2.0)

  assert profile.straight_sample_count > 0


def test_lateral_profile_filters_driver_and_lane_change_samples():
  msgs = []
  for i in range(6):
    t = float(i)
    msgs.extend([
      msg("carState", t, vEgo=16.0, steeringPressed=i < 2, leftBlinker=False, rightBlinker=False, steeringAngleDeg=0.1),
      msg("carControl", t, latActive=True),
      msg("carOutput", t, actuatorsOutput=SimpleNamespace(torque=0.0)),
      msg("modelV2", t, meta=SimpleNamespace(laneChangeState=FakeEnum("laneChangeStarting" if i >= 2 else "off"))),
      controls_msg(t, raw_curvature=0.0005),
    ])

  profile = build_lateral_oscillation_profile(msgs, source="filtered")

  assert profile.straight_sample_count == 0
  assert profile.straight_candidate_percent == 0.0


def test_lateral_profile_can_skip_sort_for_ordered_messages(monkeypatch):
  msgs = [
    msg("carState", 0.0, vEgo=15.0, steeringPressed=False, leftBlinker=False, rightBlinker=False, steeringAngleDeg=0.0),
    msg("carControl", 0.0, latActive=True),
    msg("carOutput", 0.0, actuatorsOutput=SimpleNamespace(torque=0.0)),
    msg("modelV2", 0.0, meta=SimpleNamespace(laneChangeState=FakeEnum("off"))),
    controls_msg(0.0, raw_curvature=0.0001),
  ]

  def fail_if_sorted(*args, **kwargs):
    raise AssertionError("ordered profile input should not be sorted again")

  monkeypatch.setattr(lateral_oscillation_profile, "sorted", fail_if_sorted, raising=False)

  profile = build_lateral_oscillation_profile(msgs, source="ordered", already_sorted=True)

  assert profile.source == "ordered"
