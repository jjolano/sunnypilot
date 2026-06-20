from __future__ import annotations

from openpilot.sunnypilot.system.params_migration import LATERAL_DEMAND_DEFAULT_OFF_MIGRATION_VERSION, run_migration


class FakeParams:
  def __init__(self):
    self.values = {
      "CustomLateralDemandEnabled": True,
      "CurveMemoryEnabled": True,
      "CustomLateralDemandDefaultOffMigrated": None,
    }

  def get(self, key, return_default=False):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def put(self, key, value, block=False):
    self.values[key] = value

  def put_bool(self, key, value, block=False):
    self.values[key] = bool(value)


def test_lateral_demand_default_off_migration_resets_existing_default_on_values():
  params = FakeParams()

  run_migration(params)

  assert params.values["CustomLateralDemandEnabled"] is False
  assert params.values["CurveMemoryEnabled"] is False
  assert params.values["CustomLateralDemandDefaultOffMigrated"] == LATERAL_DEMAND_DEFAULT_OFF_MIGRATION_VERSION


def test_lateral_demand_default_off_migration_is_one_shot():
  params = FakeParams()
  params.values["CustomLateralDemandDefaultOffMigrated"] = LATERAL_DEMAND_DEFAULT_OFF_MIGRATION_VERSION

  run_migration(params)

  assert params.values["CustomLateralDemandEnabled"] is True
  assert params.values["CurveMemoryEnabled"] is True
