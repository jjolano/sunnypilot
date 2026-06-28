"""Unit tests for OscillationObserver.

The observer is shadow-only: it must only log a classification and must never affect
control. Tests focus on gating, sign-reversal counting, and conservative fail-closed
behavior.
"""
from __future__ import annotations

import math

import pytest

from openpilot.sunnypilot.custom.lateral.oscillation_observer import (
  OscillationObserver,
  OSCILLATION_NONE, OSCILLATION_MILD, OSCILLATION_MODERATE, OSCILLATION_SEVERE,
)

DT = 0.01


def make_observer(dt: float = DT) -> OscillationObserver:
  return OscillationObserver(dt)


def update(observer: OscillationObserver, *, active: bool = True, v_ego: float = 20.0,
           steering_pressed: bool = False, steer_limited_by_safety: bool = False,
           curvature_limited: bool = False, output_torque: float = 0.0,
           steer_max: float = 3.0, desired_lateral_accel: float = 0.0,
           actual_lateral_accel: float = 0.0, steering_rate_deg: float = 0.0) -> int:
  debug = observer.update(
    active=active,
    v_ego=v_ego,
    steering_pressed=steering_pressed,
    steer_limited_by_safety=steer_limited_by_safety,
    curvature_limited=curvature_limited,
    output_torque=output_torque,
    steer_max=steer_max,
    desired_lateral_accel=desired_lateral_accel,
    actual_lateral_accel=actual_lateral_accel,
    steering_rate_deg=steering_rate_deg,
  )
  return int(debug.classification)


def feed_alternating(observer: OscillationObserver, periods: float, amp: float = 0.18,
                      desired_lateral_accel: float = 0.0,
                      *, output_amp: float | None = None, half_period_s: float = 0.20) -> None:
  """Feed alternating-sign output torque and lateral accel error in square-wave steps."""
  output_amp = output_amp if output_amp is not None else amp
  steps = int(periods / DT)
  half_period_steps = max(1, int(half_period_s / DT))
  for i in range(steps):
    sign = 1.0 if (i // half_period_steps) % 2 == 0 else -1.0
    actual = desired_lateral_accel + sign * amp
    output_torque = sign * output_amp
    update(observer, output_torque=output_torque, actual_lateral_accel=actual,
           desired_lateral_accel=desired_lateral_accel)


def test_inactive_returns_none_and_clears_history():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, active=False) == OSCILLATION_NONE
  assert len(obs.torque_history) == 0
  assert len(obs.error_history) == 0


def test_repeated_alternating_near_straight_classifies():
  obs = make_observer()
  feed_alternating(obs, 5)
  result = update(obs)
  assert result in (OSCILLATION_MILD, OSCILLATION_MODERATE, OSCILLATION_SEVERE)


def test_slow_repeated_alternating_near_straight_classifies():
  obs = make_observer()
  feed_alternating(obs, 6, half_period_s=1.0)
  assert update(obs) in (OSCILLATION_MILD, OSCILLATION_MODERATE)


def test_insufficient_reversals_returns_none():
  obs = make_observer()
  feed_alternating(obs, 0.12)  # less than one alternation
  assert update(obs) == OSCILLATION_NONE


def test_non_alternating_output_returns_none():
  obs = make_observer()
  for _ in range(120):
    update(obs, output_torque=0.6, actual_lateral_accel=0.05 * math.sin(_ / 10.0))
  assert update(obs) == OSCILLATION_NONE


def test_high_desired_lateral_accel_returns_none():
  obs = make_observer()
  feed_alternating(obs, 5, desired_lateral_accel=0.8)
  assert update(obs) == OSCILLATION_NONE


def test_large_tracking_error_returns_none():
  obs = make_observer()
  for _ in range(120):
    sign = 1.0 if (_ // 10) % 2 == 0 else -1.0
    update(obs, output_torque=sign * 0.6, actual_lateral_accel=sign * 2.0)
  assert update(obs) == OSCILLATION_NONE


def test_steering_pressed_returns_none():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, steering_pressed=True) == OSCILLATION_NONE


def test_steer_limited_returns_none():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, steer_limited_by_safety=True) == OSCILLATION_NONE


def test_curvature_limited_returns_none():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, curvature_limited=True) == OSCILLATION_NONE


def test_low_speed_returns_none():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, v_ego=5.0) == OSCILLATION_NONE


def test_high_steer_rate_returns_none():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, steering_rate_deg=150.0) == OSCILLATION_NONE


def test_nonfinite_input_resets():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, v_ego=math.nan) == OSCILLATION_NONE
  assert len(obs.torque_history) == 0
  assert obs.last_debug.gated is True


def test_current_large_tracking_error_clears_history():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, actual_lateral_accel=0.7) == OSCILLATION_NONE
  assert len(obs.torque_history) == 0
  assert len(obs.error_history) == 0
  assert obs.last_debug.gated is True


def test_zero_steer_max_returns_none():
  obs = make_observer()
  feed_alternating(obs, 4)
  assert update(obs, steer_max=0.0) == OSCILLATION_NONE


def test_severity_increases_with_more_reversals():
  obs = make_observer()
  feed_alternating(obs, 2)
  mild = update(obs)
  feed_alternating(obs, 3)
  moderate = update(obs)
  feed_alternating(obs, 4)
  severe = update(obs)
  assert mild <= moderate <= severe
  assert severe == OSCILLATION_SEVERE


@pytest.mark.parametrize("invalid_active,invalid_gate", [
  (False, True),
  (True, False),
])
def test_gated_flag_reflects_invalid_context(invalid_active, invalid_gate):
  obs = make_observer()
  obs.update(
    active=invalid_active,
    v_ego=20.0,
    steering_pressed=False,
    steer_limited_by_safety=False,
    curvature_limited=False,
    output_torque=0.0,
    steer_max=3.0,
    desired_lateral_accel=0.0,
    actual_lateral_accel=0.0,
    steering_rate_deg=0.0,
  )
  assert obs.last_debug.gated is invalid_gate
