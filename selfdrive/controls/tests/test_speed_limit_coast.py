import pytest
from cereal import custom

from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_comfort import (
  SPEED_LIMIT_GENTLE_ACCEL_MAX,
  apply_speed_limit_comfort_accel,
  should_apply_speed_limit_comfort_accel,
)


LongitudinalPlanSourceSP = custom.LongitudinalPlanSP.LongitudinalPlanSource


def test_speed_limit_overspeed_uses_coast_floor():
  accel = apply_speed_limit_comfort_accel(22.0, 20.0, -0.3, -1.0)

  assert accel == pytest.approx(-0.3)


def test_speed_limit_downhill_overspeed_uses_downhill_coast_floor():
  accel = apply_speed_limit_comfort_accel(22.0, 20.0, 0.25, -1.0)

  assert accel == pytest.approx(0.25)


def test_speed_limit_higher_target_caps_accel():
  accel = apply_speed_limit_comfort_accel(15.0, 20.0, -0.3, 1.0)

  assert accel == pytest.approx(SPEED_LIMIT_GENTLE_ACCEL_MAX)


def test_speed_limit_higher_target_preserves_lower_accel():
  accel = apply_speed_limit_comfort_accel(15.0, 20.0, -0.3, 0.2)

  assert accel == pytest.approx(0.2)


def test_speed_limit_comfort_gating_requires_speed_limit_source_without_safety_overrides():
  speed_limit = LongitudinalPlanSourceSP.speedLimitAssist
  cruise = LongitudinalPlanSourceSP.cruise

  assert should_apply_speed_limit_comfort_accel(False, False, False, False, False, speed_limit)
  assert not should_apply_speed_limit_comfort_accel(True, False, False, False, False, speed_limit)
  assert not should_apply_speed_limit_comfort_accel(False, True, False, False, False, speed_limit)
  assert not should_apply_speed_limit_comfort_accel(False, False, True, False, False, speed_limit)
  assert not should_apply_speed_limit_comfort_accel(False, False, False, True, False, speed_limit)
  assert not should_apply_speed_limit_comfort_accel(False, False, False, False, True, speed_limit)
  assert not should_apply_speed_limit_comfort_accel(False, False, False, False, False, cruise)
