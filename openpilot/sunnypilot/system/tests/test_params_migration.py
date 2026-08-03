from __future__ import annotations

import pytest

from openpilot.sunnypilot.system.params_migration import (
  LATERAL_DEMAND_DEFAULT_OFF_MIGRATION_VERSION,
  LONGITUDINAL_MODE_MIGRATION_VERSION,
  run_migration,
)
from opendbc.sunnypilot.car.tesla.values import MadsScreenButtonType


class FakeParams:
  def __init__(self, **initial):
    self.values = dict(initial)

  def get(self, key, return_default=False):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put(self, key, value, block=False):
    self.values[key] = value

  def put_bool(self, key, value, block=False):
    self.values[key] = bool(value)


def test_lateral_demand_default_off_migration_resets_existing_default_on_values():
  params = FakeParams(CustomLateralDemandEnabled=True, CustomLateralDemandDefaultOffMigrated=None)

  run_migration(params)

  assert params.values["CustomLateralDemandEnabled"] is False
  assert params.values["CustomLateralDemandDefaultOffMigrated"] == LATERAL_DEMAND_DEFAULT_OFF_MIGRATION_VERSION


def test_lateral_demand_default_off_migration_is_one_shot():
  params = FakeParams(CustomLateralDemandEnabled=True, CustomLateralDemandDefaultOffMigrated=LATERAL_DEMAND_DEFAULT_OFF_MIGRATION_VERSION)

  run_migration(params)

  assert params.values["CustomLateralDemandEnabled"] is True


def test_longitudinal_mode_migration_from_legacy_int_acc():
  params = FakeParams(LongitudinalMode="0", LongitudinalModeMigrationVersion=None)
  run_migration(params)
  assert params.values["CustomLongitudinalMode"] == "acc"
  assert params.values["LongitudinalModeMigrationVersion"] == LONGITUDINAL_MODE_MIGRATION_VERSION


def test_longitudinal_mode_migration_from_legacy_int_e2e():
  params = FakeParams(LongitudinalMode="1", LongitudinalModeMigrationVersion=None)
  run_migration(params)
  assert params.values["CustomLongitudinalMode"] == "e2e"


def test_longitudinal_mode_migration_from_legacy_int_scc():
  params = FakeParams(LongitudinalMode="2", LongitudinalModeMigrationVersion=None)
  run_migration(params)
  assert params.values["CustomLongitudinalMode"] == "scc"


def test_longitudinal_mode_migration_from_experimental_and_dec():
  params = FakeParams(ExperimentalMode=True, DynamicExperimentalControl=True, LongitudinalModeMigrationVersion=None)
  run_migration(params)
  assert params.values["CustomLongitudinalMode"] == "scc"


def test_longitudinal_mode_migration_from_experimental_only():
  params = FakeParams(ExperimentalMode=True, DynamicExperimentalControl=False, LongitudinalModeMigrationVersion=None)
  run_migration(params)
  assert params.values["CustomLongitudinalMode"] == "e2e"


def test_longitudinal_mode_migration_fresh_install_does_not_write_mode():
  params = FakeParams(LongitudinalModeMigrationVersion=None)
  run_migration(params)
  assert "CustomLongitudinalMode" not in params.values
  assert params.values["LongitudinalModeMigrationVersion"] == LONGITUDINAL_MODE_MIGRATION_VERSION


def test_longitudinal_mode_migration_respects_existing_choice():
  params = FakeParams(CustomLongitudinalMode="acc", LongitudinalMode="2", LongitudinalModeMigrationVersion=None)
  run_migration(params)
  assert params.values["CustomLongitudinalMode"] == "acc"


def test_longitudinal_mode_migration_is_one_shot():
  params = FakeParams(LongitudinalMode="0", LongitudinalModeMigrationVersion=LONGITUDINAL_MODE_MIGRATION_VERSION)
  run_migration(params)
  assert "CustomLongitudinalMode" not in params.values


def test_longitudinal_mode_migration_upgrades_from_legacy_version():
  params = FakeParams(LongitudinalMode="1", LongitudinalModeMigrationVersion="1.1")
  run_migration(params)
  assert params.values["CustomLongitudinalMode"] == "e2e"
  assert params.values["LongitudinalModeMigrationVersion"] == LONGITUDINAL_MODE_MIGRATION_VERSION


def test_tesla_mads_screen_button_migration_preserves_existing_value():
  params = FakeParams(CarPlatformBundle={"brand": "tesla"}, TeslaMadsScreenButton=MadsScreenButtonType.FOUR_FINGER)
  run_migration(params)
  assert params.values["TeslaMadsScreenButton"] == MadsScreenButtonType.FOUR_FINGER


def test_tesla_mads_screen_button_migration_seeds_existing_tesla():
  params = FakeParams(CarPlatformBundle={"brand": "tesla"})
  run_migration(params)
  assert params.values["TeslaMadsScreenButton"] == MadsScreenButtonType.THREE_FINGER


@pytest.mark.parametrize("initial", [{"CarPlatformBundle": {"brand": "toyota"}}, {}])
def test_tesla_mads_screen_button_migration_leaves_non_tesla_and_fresh_unset(initial):
  params = FakeParams(**initial)
  run_migration(params)
  assert "TeslaMadsScreenButton" not in params.values


def test_tesla_mads_screen_button_migration_ignores_malformed_car_params():
  params = FakeParams(CarParamsPersistent=b"not-car-params")
  run_migration(params)
  assert "TeslaMadsScreenButton" not in params.values
