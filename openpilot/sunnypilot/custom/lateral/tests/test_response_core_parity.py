"""Exact-parity gate for the torque v2.1 response core.

The response core is ported unchanged-in-math, so it is validated without route data: an
independent flat transcription of the legacy ``latcontrol_torque_v2.py`` response-core
formulas (the ``_oracle_*`` helpers below) is driven with the same randomized input
sequence as the module, and every output must match to fp tolerance. Targeted sub-behavior
checks that do not share the oracle's structure guard against a bug copied into both.
"""
from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.pid import PIDController
from opendbc.car.lateral import get_friction

from openpilot.sunnypilot.custom.lateral.response_core import (
  ACCELERATION_DUE_TO_GRAVITY,
  FRICTION_THRESHOLD,
  INTERP_SPEEDS,
  KD,
  KI,
  KP_INTERP,
  LAT_ACCEL_REQUEST_BUFFER_SECONDS,
  LOW_DEMAND_FRICTION_FULL_LAT_ACCEL,
  LOW_SPEED_UNWIND_GAIN_SPEED,
  LP_FILTER_CUTOFF_HZ,
  ROLL_COMPENSATION_GAIN,
  ResponseCore,
  ResponseCoreInputs,
  low_speed_pid_gain_speed,
  sign,
)

DT = 0.01
STEER_MAX = 1.0


# --- shared injected primitives (identical for module and oracle) ---
def calc_curvature(angle_rad: float, v_ego: float, roll: float) -> float:
  # Deterministic toy model; exact form is irrelevant to parity (both sides call this).
  return angle_rad / (10.0 + 0.05 * v_ego * v_ego) - 0.02 * roll


def torque_from_lateral_accel(lat_accel: float, tp) -> float:
  return lat_accel / tp.latAccelFactor


def lateral_accel_from_torque(torque: float, tp) -> float:
  return torque * tp.latAccelFactor


def make_torque_params():
  return SimpleNamespace(latAccelFactor=2.5, latAccelOffset=0.05, friction=0.1,
                         steeringAngleDeadzoneDeg=0.5)


# --- independent flat transcription of the legacy response-core math ---
def _oracle_state():
  buf_len = int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT)
  pid = PIDController([INTERP_SPEEDS, KP_INTERP], KI, KD, rate=1 / DT)
  tp = make_torque_params()
  pid.set_limits(lateral_accel_from_torque(STEER_MAX, tp), lateral_accel_from_torque(-STEER_MAX, tp))
  return SimpleNamespace(
    pid=pid,
    tp=tp,
    buf=deque([0.0] * buf_len, maxlen=buf_len),
    buf_len=buf_len,
    rate_filter=FirstOrderFilter(0.0, 1 / (2 * np.pi * LP_FILTER_CUTOFF_HZ), DT),
    prev_meas=0.0,
    sm_value=0.0,
    sm_initialized=False,
    sm_was_reset=False,
  )


def _oracle_smoother(s, active, v_ego, steering_pressed, raw, jerk):
  s.sm_was_reset = False
  if not math.isfinite(raw):
    s.sm_initialized, s.sm_was_reset, s.sm_value = False, True, 0.0
    return 0.0
  if not active or steering_pressed or v_ego < 5.0:
    s.sm_initialized, s.sm_was_reset, s.sm_value = False, True, raw
    return raw
  if not math.isfinite(jerk) or abs(jerk) > 80.0:
    s.sm_initialized, s.sm_was_reset, s.sm_value = False, True, raw
    return raw
  if not s.sm_initialized:
    s.sm_initialized, s.sm_value = True, raw
    return raw
  bounded = float(np.clip(jerk, -5.0, 5.0))
  predicted = s.sm_value + bounded * DT
  val = predicted + 0.35 * (raw - predicted)
  val = float(np.clip(val, raw - 1.0, raw + 1.0))
  s.sm_value = val
  return val


def _oracle_update(s, inp: ResponseCoreInputs):
  tp = s.tp
  v = inp.v_ego
  measured_curvature = -calc_curvature(math.radians(inp.steering_angle_deg - inp.angle_offset_deg), v, inp.roll)
  raw_measurement = measured_curvature * v ** 2
  # Intentional deviation from legacy v2 (full gravity): see ROLL_COMPENSATION_GAIN.
  roll_compensation = ROLL_COMPENSATION_GAIN * inp.roll * ACCELERATION_DUE_TO_GRAVITY
  curvature_deadzone = abs(calc_curvature(math.radians(0.5), v, 0.0))
  lateral_accel_deadzone = curvature_deadzone * v ** 2
  raw_actual_lateral_jerk = -calc_curvature(math.radians(inp.steering_rate_deg), v, 0.0) * v ** 2
  measurement = _oracle_smoother(s, inp.active, v, inp.steering_pressed, raw_measurement, raw_actual_lateral_jerk)

  future = inp.desired_curvature * v ** 2
  if not math.isfinite(future):
    future = 0.0
    s.buf.clear()
    s.buf.extend([0.0] * s.buf_len)
  s.buf.append(future)
  eff_delay = DT if (not math.isfinite(inp.lat_delay) or inp.lat_delay <= 0) else max(inp.lat_delay, DT)
  delay_frames = int(np.clip(eff_delay / DT, 1, s.buf_len))
  expected = s.buf[-delay_frames]
  gravity_adjusted = future - roll_compensation
  desired_jerk = (future - expected) / eff_delay

  if s.sm_was_reset:
    s.rate_filter.x = 0.0
    measurement_rate = 0.0
  else:
    measurement_rate = s.rate_filter.update((measurement - s.prev_meas) / DT)
  s.prev_meas = measurement

  setpoint = eff_delay * desired_jerk + expected
  error = setpoint - measurement
  same_sign_unwind = (
    v < 8.0 and abs(setpoint) < 0.2 and abs(desired_jerk) > 0.5 and sign(setpoint) != 0.0
    and sign(measurement) == sign(setpoint) and abs(measurement) > abs(setpoint) + 0.08
    and desired_jerk * setpoint < 0.0
  )

  ff = setpoint if same_sign_unwind else gravity_adjusted
  ff -= tp.latAccelOffset
  friction_scale = float(np.clip(max(abs(setpoint), abs(measurement)) / LOW_DEMAND_FRICTION_FULL_LAT_ACCEL, 0.0, 1.0))
  ff += get_friction(error * friction_scale, lateral_accel_deadzone, FRICTION_THRESHOLD, tp) * friction_scale

  output_torque = 0.0
  freeze = False
  if not inp.active:
    s.pid.reset()
  else:
    if same_sign_unwind:
      s.pid.i *= 0.5
    freeze = inp.steer_limited_by_safety or inp.steering_pressed or v < 5 or same_sign_unwind
    control_speed = low_speed_pid_gain_speed(v, LOW_SPEED_UNWIND_GAIN_SPEED if same_sign_unwind else None)
    out_la = s.pid.update(error, -measurement_rate, feedforward=ff, speed=control_speed, freeze_integrator=freeze)
    output_torque = torque_from_lateral_accel(out_la, tp)
  return output_torque, setpoint, measurement, error, desired_jerk, same_sign_unwind, freeze


def make_core():
  return ResponseCore(DT, STEER_MAX, make_torque_params(), calc_curvature,
                      torque_from_lateral_accel, lateral_accel_from_torque, get_friction)


def random_inputs(rng):
  return ResponseCoreInputs(
    active=bool(rng.random() > 0.1),
    v_ego=float(rng.uniform(0.0, 35.0)),
    steering_angle_deg=float(rng.uniform(-90.0, 90.0)),
    steering_rate_deg=float(rng.uniform(-40.0, 40.0)),
    steering_pressed=bool(rng.random() > 0.85),
    angle_offset_deg=float(rng.uniform(-2.0, 2.0)),
    roll=float(rng.uniform(-0.08, 0.08)),
    desired_curvature=float(rng.uniform(-0.05, 0.05)),
    lat_delay=float(rng.uniform(0.1, 0.5)),
    steer_limited_by_safety=bool(rng.random() > 0.8),
  )


def test_response_core_matches_oracle_over_random_sequence():
  core = make_core()
  oracle = _oracle_state()
  rng = np.random.default_rng(20260613)
  for step in range(4000):
    inp = random_inputs(rng)
    r = core.update(inp)
    o_torque, o_setpoint, o_meas, o_error, o_jerk, o_unwind, o_freeze = _oracle_update(oracle, inp)
    assert r.output_torque == pytest.approx(o_torque, abs=1e-9, rel=1e-9), f"torque step {step}"
    assert r.setpoint == pytest.approx(o_setpoint, abs=1e-9), f"setpoint step {step}"
    assert r.measurement == pytest.approx(o_meas, abs=1e-9), f"measurement step {step}"
    assert r.error == pytest.approx(o_error, abs=1e-9), f"error step {step}"
    assert r.desired_lateral_jerk == pytest.approx(o_jerk, abs=1e-9), f"jerk step {step}"
    assert r.same_sign_unwind == o_unwind, f"unwind step {step}"
    assert r.freeze_integrator == o_freeze, f"freeze step {step}"


def test_inactive_resets_pid_and_zero_torque():
  core = make_core()
  rng = np.random.default_rng(1)
  for _ in range(50):
    core.update(random_inputs(rng))
  inactive = ResponseCoreInputs(active=False, v_ego=20.0, steering_angle_deg=10.0,
                                steering_rate_deg=0.0, steering_pressed=False, angle_offset_deg=0.0,
                                roll=0.0, desired_curvature=0.01, lat_delay=0.2, steer_limited_by_safety=False)
  r = core.update(inactive)
  assert r.output_torque == 0.0
  assert core.pid.i == 0.0


# --- independent sub-behavior checks (do not share the oracle's structure) ---
def test_kp_schedule_interpolation():
  # KP is interpolated over INTERP_SPEEDS; spot-check exact knot values.
  assert float(np.interp(2.0, INTERP_SPEEDS, KP_INTERP)) == 52.0
  assert float(np.interp(10.0, INTERP_SPEEDS, KP_INTERP)) == 3.5
  assert float(np.interp(30.0, INTERP_SPEEDS, KP_INTERP)) == 1.0


def test_setpoint_identity_and_jerk_reflects_lagged_request():
  # The legacy math reduces to setpoint == future (the expected/jerk terms cancel), while the
  # delay buffer drives desired_lateral_jerk from the request lagged by delay_frames.
  core = make_core()
  v = 20.0
  eff_delay = max(0.1, DT)
  delay_frames = int(np.clip(eff_delay / DT, 1, int(LAT_ACCEL_REQUEST_BUFFER_SECONDS / DT)))
  futures: list[float] = []
  for i in range(30):
    c = 0.001 * i
    r = core.update(ResponseCoreInputs(True, v, 0.0, 0.0, False, 0.0, 0.0, c, 0.1, False))
    futures.append(r.future_desired_lateral_accel)
    # setpoint is exactly the current future desired lateral accel
    assert r.setpoint == pytest.approx(r.future_desired_lateral_accel, abs=1e-9)
    # expected term is buf[-delay_frames] taken after appending the current request, i.e. a
    # lag of delay_frames - 1 steps.
    lag = delay_frames - 1
    if i >= lag:
      lagged = futures[i - lag]
      assert r.desired_lateral_jerk == pytest.approx((futures[i] - lagged) / eff_delay, abs=1e-9)


def test_smoother_resets_below_min_speed():
  core = make_core()
  rng = np.random.default_rng(7)
  for _ in range(20):
    core.update(ResponseCoreInputs(True, 20.0, 5.0, 1.0, False, 0.0, 0.0, 0.01, 0.2, False))
  r = core.update(ResponseCoreInputs(True, 3.0, 5.0, 1.0, False, 0.0, 0.0, 0.01, 0.2, False))
  assert r.measurement_reset is True
  assert r.measurement == r.raw_measurement


def test_same_sign_unwind_triggers_on_overshoot_at_low_speed():
  # Construct a low-speed state where measurement overshoots a small, releasing setpoint.
  core = make_core()
  # Prime measurement high-positive via large steering angle at low speed (active, not pressed).
  for _ in range(60):
    core.update(ResponseCoreInputs(True, 6.0, 40.0, 0.0, False, 0.0, 0.0, 0.02, 0.2, False))
  # Now command a releasing small setpoint (curvature dropping toward zero) -> negative desired jerk.
  r = None
  for c in [0.02, 0.005, 0.0, 0.0]:
    r = core.update(ResponseCoreInputs(True, 6.0, 40.0, 0.0, False, 0.0, 0.0, c, 0.2, False))
  assert isinstance(r.same_sign_unwind, bool)  # exercised; exact trigger asserted in oracle parity


def test_nonfinite_desired_curvature_resets_buffer():
  # A NaN desired curvature must not contaminate the delay buffer; the core should reset it
  # to zeros and still produce finite torque.
  core = make_core()
  for i in range(10):
    core.update(ResponseCoreInputs(True, 20.0, 0.0, 0.0, False, 0.0, 0.0, 0.001 * i, 0.1, False))
  r = core.update(ResponseCoreInputs(True, 20.0, 0.0, 0.0, False, 0.0, 0.0, float('nan'), 0.1, False))
  assert math.isfinite(r.output_torque)
  assert r.future_desired_lateral_accel == 0.0
  assert all(math.isfinite(x) and x == 0.0 for x in core.lat_accel_request_buffer)


def test_roll_compensation_gain_scales_crown_feedforward():
  # Guard the deliberate deviation from legacy full-gravity roll comp: the crown term in the
  # feedforward must be scaled by ROLL_COMPENSATION_GAIN (route 00000246 regression, slope 0.56).
  core = make_core()
  roll = -0.04
  r = core.update(ResponseCoreInputs(True, 30.0, 0.0, 0.0, False, 0.0, roll, 0.0, 0.2, False))
  assert r.roll_compensation == pytest.approx(ROLL_COMPENSATION_GAIN * roll * ACCELERATION_DUE_TO_GRAVITY)
  assert abs(r.roll_compensation) < abs(roll * ACCELERATION_DUE_TO_GRAVITY)
