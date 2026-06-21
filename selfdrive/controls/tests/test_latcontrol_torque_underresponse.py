import pytest

from cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.sunnypilot.selfdrive.controls.lib.underresponse_sentinel import (
  BLOCK_CURVATURE_LIMITED,
  BLOCK_DESIRED_NOT_PERSISTENT,
  BLOCK_ERROR_TOO_SMALL,
  BLOCK_FAST_CLOSING,
  BLOCK_INACTIVE,
  BLOCK_LOW_SPEED,
  BLOCK_ROLL_TOO_HIGH,
  BLOCK_ROLL_UNSTABLE,
  BLOCK_STEER_LIMITED,
  BLOCK_STEERING_PRESSED,
  BLOCK_TORQUE_SATURATED,
  UnderresponseDebug,
  UnderresponseSentinel,
  write_underresponse_debug,
)


def base_inputs():
  return {
    "active": True,
    "v_ego": 25.0,
    "steering_pressed": False,
    "steer_limited_by_safety": False,
    "curvature_limited": False,
    "setpoint": 1.2,
    "measurement": 0.60,
    "lateral_accel_deadzone": 0.0,
    "output_torque": 0.30,
    "steer_max": 1.0,
    "roll": 0.0,
  }


def update_with(sentinel, **overrides):
  args = base_inputs()
  args.update(overrides)
  return sentinel.update(**args)


def run_frames(sentinel, frames=120, **overrides):
  debug = None
  for _ in range(frames):
    debug = update_with(sentinel, **overrides)
  assert debug is not None
  return debug


def test_triggers_on_persistent_underresponse():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s)
  assert dbg.active
  assert dbg.eligible
  assert dbg.block_mask == 0
  assert dbg.error > 0.0
  assert dbg.error_filtered > 0.25
  assert dbg.duration >= 0.20
  assert dbg.shadow_lat_accel > 0.0
  assert 0.0 <= dbg.severity <= 1.0


def test_inactive_resets_after_trigger():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s)
  assert dbg.active
  dbg = update_with(s, active=False)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_INACTIVE
  dbg = update_with(s)
  assert not dbg.active


def test_no_trigger_low_speed():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s, v_ego=5.0)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_LOW_SPEED


def test_no_trigger_steering_pressed():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s, steering_pressed=True)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_STEERING_PRESSED


def test_no_trigger_steer_limited_by_safety():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s, steer_limited_by_safety=True)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_STEER_LIMITED


def test_no_trigger_curvature_limited():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s, curvature_limited=True)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_CURVATURE_LIMITED


def test_no_trigger_near_output_torque_cap():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s, output_torque=0.93, steer_max=1.0)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_TORQUE_SATURATED


def test_no_trigger_high_roll():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s, roll=0.12)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_ROLL_TOO_HIGH


def test_no_trigger_unstable_roll():
  s = UnderresponseSentinel(DT_CTRL)
  saw_unstable = False
  dbg = None
  for i in range(120):
    dbg = update_with(s, roll=0.0 if i % 2 == 0 else 0.01)
    saw_unstable = saw_unstable or bool(dbg.block_mask & BLOCK_ROLL_UNSTABLE)
  assert dbg is not None
  assert not dbg.active
  assert saw_unstable


def test_no_trigger_desired_sign_flip():
  s = UnderresponseSentinel(DT_CTRL)
  saw_not_persistent = False
  dbg = None
  for i in range(120):
    sign = 1.0 if (i // 5) % 2 == 0 else -1.0
    dbg = update_with(s, setpoint=sign * 1.2, measurement=sign * 0.60)
    saw_not_persistent = saw_not_persistent or bool(dbg.block_mask & BLOCK_DESIRED_NOT_PERSISTENT)
  assert dbg is not None
  assert not dbg.active
  assert saw_not_persistent


def test_no_trigger_desired_changes_too_fast():
  s = UnderresponseSentinel(DT_CTRL)
  saw_not_persistent = False
  dbg = None
  for i in range(120):
    setpoint = 1.0 if i % 2 == 0 else 1.1
    dbg = update_with(s, setpoint=setpoint, measurement=0.40)
    saw_not_persistent = saw_not_persistent or bool(dbg.block_mask & BLOCK_DESIRED_NOT_PERSISTENT)
  assert dbg is not None
  assert not dbg.active
  assert saw_not_persistent


def test_no_trigger_fast_closing():
  s = UnderresponseSentinel(DT_CTRL)
  saw_fast_closing = False
  dbg = None
  for i in range(120):
    measurement = min(1.15, 0.20 + 0.02 * i)
    dbg = update_with(s, setpoint=1.2, measurement=measurement)
    saw_fast_closing = saw_fast_closing or bool(dbg.block_mask & BLOCK_FAST_CLOSING)
  assert dbg is not None
  assert not dbg.active
  assert saw_fast_closing


def test_no_trigger_error_too_small():
  s = UnderresponseSentinel(DT_CTRL)
  dbg = run_frames(s, setpoint=1.2, measurement=1.05)
  assert not dbg.active
  assert dbg.block_mask & BLOCK_ERROR_TOO_SMALL


def test_write_underresponse_debug_to_log():
  pid_log = log.ControlsState.LateralTorqueState.new_message()
  dbg = UnderresponseDebug(
    active=True,
    eligible=True,
    block_mask=123,
    error=0.4,
    error_filtered=0.3,
    duration=0.25,
    closing_rate=-0.1,
    shadow_lat_accel=0.15,
    severity=0.7,
  )
  write_underresponse_debug(pid_log, dbg)
  assert pid_log.underresponseActive
  assert pid_log.underresponseEligible
  assert pid_log.underresponseBlockMask == 123
  assert pid_log.underresponseError == pytest.approx(0.4)
  assert pid_log.underresponseErrorFiltered == pytest.approx(0.3)
  assert pid_log.underresponseDuration == pytest.approx(0.25)
  assert pid_log.underresponseClosingRate == pytest.approx(-0.1)
  assert pid_log.underresponseShadowLatAccel == pytest.approx(0.15)
  assert pid_log.underresponseSeverity == pytest.approx(0.7)
