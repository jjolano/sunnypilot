import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_CURVATURE,
  MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
  MAX_LATERAL_ACCEL_NO_ROLL,
  clip_curvature,
  update_lateral_accel_limit,
)


def test_clip_curvature_uses_default_lateral_accel_limit():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited = clip_curvature(v_ego, requested_curvature, requested_curvature, 0.0)

  assert clipped_curvature == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL / v_ego**2)
  assert limited


def test_clip_curvature_allows_driver_gas_lateral_accel_limit():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited = clip_curvature(
    v_ego,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(requested_curvature)
  assert not limited


def test_clip_curvature_driver_gas_still_respects_max_curvature():
  requested_curvature = MAX_CURVATURE + 0.1

  clipped_curvature, limited = clip_curvature(
    1.0,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(MAX_CURVATURE)
  assert limited


def test_lateral_accel_limit_enters_driver_gas_override():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=True,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL)


def test_lateral_accel_limit_decays_after_driver_gas_release():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    dt=0.5,
  )

  expected = MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - (
    (MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - MAX_LATERAL_ACCEL_NO_ROLL) / 1.25
  ) * 0.5
  assert limit == pytest.approx(expected)


def test_lateral_accel_limit_decay_clamps_at_default():
  limit = update_lateral_accel_limit(
    3.1,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    dt=1.0,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)


@pytest.mark.parametrize(
  "lat_active,brake_pressed,steering_pressed",
  [
    (False, False, False),
    (True, True, False),
    (True, False, True),
  ],
)
def test_lateral_accel_limit_resets_for_inactive_or_driver_intervention(lat_active, brake_pressed, steering_pressed):
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
    manual_gas_override=False,
    lat_active=lat_active,
    brake_pressed=brake_pressed,
    steering_pressed=steering_pressed,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)


@pytest.mark.parametrize(
  "brake_pressed,steering_pressed",
  [
    (True, False),
    (False, True),
  ],
)
def test_lateral_accel_limit_blocks_driver_gas_override_during_driver_intervention(brake_pressed, steering_pressed):
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=True,
    lat_active=True,
    brake_pressed=brake_pressed,
    steering_pressed=steering_pressed,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)
