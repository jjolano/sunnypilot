import math

import pytest
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY

from openpilot.selfdrive.controls.lib.drive_helpers import (
  MAX_CURVATURE,
  MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
  MAX_LATERAL_ACCEL_NO_ROLL,
  clip_curvature,
  curv_from_psis,
  get_accel_from_plan,
  get_curvature_from_plan,
  should_latch_lateral_accel_burst,
  update_lateral_accel_limit,
)


def test_lateral_accel_burst_latch_allows_active_saturation():
  assert should_latch_lateral_accel_burst(
    default_lateral_accel_limited=True,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    manual_gas_override=False,
  )


@pytest.mark.parametrize(
  "default_lateral_accel_limited,lat_active,brake_pressed,steering_pressed,manual_gas_override",
  [
    (False, True, False, False, False),
    (True, False, False, False, False),
    (True, True, True, False, False),
    (True, True, False, True, False),
    (True, True, False, False, True),
  ],
)
def test_lateral_accel_burst_latch_blocks_inactive_driver_intervention_or_manual_gas(
  default_lateral_accel_limited, lat_active, brake_pressed, steering_pressed, manual_gas_override
):
  assert not should_latch_lateral_accel_burst(
    default_lateral_accel_limited=default_lateral_accel_limited,
    lat_active=lat_active,
    brake_pressed=brake_pressed,
    steering_pressed=steering_pressed,
    manual_gas_override=manual_gas_override,
  )


def test_clip_curvature_uses_default_lateral_accel_limit():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited, _ = clip_curvature(v_ego, requested_curvature, requested_curvature, 0.0)

  assert clipped_curvature == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL / v_ego**2)
  assert limited


def test_clip_curvature_allows_driver_gas_lateral_accel_limit():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited, _ = clip_curvature(
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

  clipped_curvature, limited, _ = clip_curvature(
    1.0,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(MAX_CURVATURE)
  assert limited


def test_clip_curvature_reports_default_lateral_accel_limit_saturation():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited, default_lateral_accel_limited = clip_curvature(
    v_ego,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL / v_ego**2)
  assert limited
  assert default_lateral_accel_limited


def test_clip_curvature_reports_no_default_saturation_when_request_fits_default_cap():
  v_ego = 10.0
  requested_curvature = 2.0 / v_ego**2

  clipped_curvature, limited, default_lateral_accel_limited = clip_curvature(
    v_ego,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(requested_curvature)
  assert not limited
  assert not default_lateral_accel_limited


def test_clip_curvature_reports_default_saturation_even_when_burst_cap_allows_request():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited, default_lateral_accel_limited = clip_curvature(
    v_ego,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(requested_curvature)
  assert not limited
  assert default_lateral_accel_limited


def test_clip_curvature_preserves_legacy_linear_roll_compensation_by_default():
  v_ego = 10.0
  roll = 0.5
  requested_curvature = 10.0 / v_ego**2

  clipped_curvature, limited, _ = clip_curvature(v_ego, requested_curvature, requested_curvature, roll)

  assert clipped_curvature == pytest.approx((MAX_LATERAL_ACCEL_NO_ROLL + roll * ACCELERATION_DUE_TO_GRAVITY) / v_ego**2)
  assert limited


def test_get_accel_from_plan_returns_safe_stop_for_length_mismatch():
  assert get_accel_from_plan([0.0], [0.0, 0.0], [0.0]) == (0.0, True)


def test_curv_from_psis_is_finite_and_get_curvature_from_plan_matches_interpolated_yaw():
  yaws = [0.0, 0.05, 0.1]
  yaw_rates = [0.01, 0.02, 0.03]
  t_idxs = [0.0, 1.0, 2.0]
  vego = 12.0
  action_t = 1.5

  psi_target = yaws[1] + (action_t - t_idxs[1]) * (yaws[2] - yaws[1]) / (t_idxs[2] - t_idxs[1])
  expected = curv_from_psis(psi_target, yaw_rates[0], vego, action_t)

  assert math.isfinite(expected)
  assert get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t) == pytest.approx(expected)


def test_clip_curvature_accurate_lateral_accel_uses_exact_roll_compensation():
  v_ego = 10.0
  roll = 0.5
  requested_curvature = 10.0 / v_ego**2

  clipped_curvature, limited, _ = clip_curvature(v_ego, requested_curvature, requested_curvature, roll, accurate_lateral_accel=True)

  assert clipped_curvature == pytest.approx((MAX_LATERAL_ACCEL_NO_ROLL + math.sin(roll) * ACCELERATION_DUE_TO_GRAVITY) / v_ego**2)
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


def test_lateral_accel_limit_decays_after_manual_gas_even_if_request_would_exceed_default_cap():
  manual_gas_limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=True,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
  )
  _, _, default_lateral_accel_limited = clip_curvature(
    10.0,
    0.04,
    0.04,
    0.0,
    manual_gas_limit,
  )
  should_latch = should_latch_lateral_accel_burst(
    default_lateral_accel_limited=default_lateral_accel_limited,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    manual_gas_override=True,
  )
  released_limit = update_lateral_accel_limit(
    manual_gas_limit,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=should_latch,
    dt=0.5,
  )

  expected = MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - (
    (MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - MAX_LATERAL_ACCEL_NO_ROLL) / 1.25
  ) * 0.5
  assert default_lateral_accel_limited
  assert not should_latch
  assert released_limit == pytest.approx(expected)


def test_lateral_accel_limit_preserves_positional_dt_argument():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
    False,
    True,
    False,
    False,
    0.5,
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


def test_lateral_accel_limit_enters_burst_after_default_cap_saturation():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=True,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL)


def test_lateral_accel_limit_does_not_burst_without_default_cap_saturation():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=False,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)


def test_lateral_accel_limit_refreshes_burst_while_default_cap_saturation_continues():
  limit = update_lateral_accel_limit(
    4.0,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=True,
    dt=0.5,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL)


def test_lateral_accel_limit_decays_after_burst_saturation_ends():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=False,
    dt=0.5,
  )

  expected = MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - (
    (MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - MAX_LATERAL_ACCEL_NO_ROLL) / 1.25
  ) * 0.5
  assert limit == pytest.approx(expected)


@pytest.mark.parametrize(
  "lat_active,brake_pressed,steering_pressed",
  [
    (False, False, False),
    (True, True, False),
    (True, False, True),
  ],
)
def test_lateral_accel_limit_resets_burst_for_inactive_or_driver_intervention(lat_active, brake_pressed, steering_pressed):
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
    manual_gas_override=False,
    lat_active=lat_active,
    brake_pressed=brake_pressed,
    steering_pressed=steering_pressed,
    default_lateral_accel_limited=True,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)


def test_lateral_accel_limit_driver_gas_wins_over_burst_state():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=True,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=False,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL)
