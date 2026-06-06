import math

from openpilot.sunnypilot.selfdrive.controls.lib.torque_v4_route_metrics import (
  TorqueV4RouteFrame,
  compute_torque_v4_route_metrics,
)


def frame(**overrides):
  values = {
    "target_lateral_accel": 0.5,
    "actual_lateral_accel": 0.4,
    "raw_torque": 0.2,
    "governed_torque": 0.18,
    "governor_reason": 0,
    "v_ego": 20.0,
  }
  values.update(overrides)
  return TorqueV4RouteFrame(**values)


def test_route_metrics_are_finite_for_empty_and_invalid_inputs():
  empty = compute_torque_v4_route_metrics(())
  invalid = compute_torque_v4_route_metrics((frame(target_lateral_accel=float("nan")),))

  for metrics in (empty, invalid):
    assert math.isfinite(metrics.target_lateral_accel_rms)
    assert math.isfinite(metrics.actual_lateral_accel_rms)
    assert math.isfinite(metrics.phase_lag_proxy)
    assert math.isfinite(metrics.overshoot)


def test_route_metrics_capture_wrong_sign_overshoot_and_governor_fractions():
  metrics = compute_torque_v4_route_metrics((
    frame(target_lateral_accel=0.5, actual_lateral_accel=-0.2, governor_reason=1),
    frame(target_lateral_accel=0.7, actual_lateral_accel=0.9, governor_reason=1, same_direction_actuator_limited=True),
    frame(target_lateral_accel=0.2, actual_lateral_accel=0.3, governor_reason=2),
  ), dt=0.1)

  assert metrics.wrong_sign_response_duration == 0.1
  assert metrics.overshoot > 0.0
  assert metrics.governor_reason_fraction[1] == 2 / 3
  assert metrics.governor_reason_fraction[2] == 1 / 3
  assert metrics.same_direction_actuator_limit_fraction == 1 / 3


def test_route_metrics_capture_phase_lag_driver_recovery_and_wander_energy():
  metrics = compute_torque_v4_route_metrics((
    frame(target_lateral_accel=0.0, actual_lateral_accel=0.1, v_ego=15.0),
    frame(target_lateral_accel=1.0, actual_lateral_accel=0.2, driver_override=True),
    frame(target_lateral_accel=0.6, actual_lateral_accel=1.1),
    frame(target_lateral_accel=0.2, actual_lateral_accel=0.25),
  ), dt=0.2)

  assert metrics.phase_lag_proxy > 0.0
  assert metrics.driver_override_recovery == 0.2
  assert metrics.straight_road_wander_energy > 0.0
