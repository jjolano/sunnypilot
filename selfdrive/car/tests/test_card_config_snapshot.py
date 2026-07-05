import pytest

from cereal import car

from openpilot.selfdrive.car.card import (
  CardConfigSnapshot,
  _read_card_config_snapshot,
)


class DummyParams:
  def __init__(self, values):
    self.values = values

  def get_bool(self, key):
    return bool(self.values.get(key, False))

  def get(self, key, return_default=False):
    return self.values.get(key)


@pytest.mark.parametrize(
  ("openpilot_longitudinal_control", "values", "expected"),
  [
    (
      True,
      {
        "IsMetric": True,
        "ExperimentalMode": True,
        "DynamicExperimentalControl": True,
      },
      CardConfigSnapshot(True, True, True),
    ),
    (
      True,
      {
        "IsMetric": False,
        "CustomLongitudinalEnabled": True,
        "CustomLongitudinalMode": "scc",
        "ExperimentalMode": False,
        "DynamicExperimentalControl": True,
      },
      CardConfigSnapshot(False, False, False),
    ),
    (
      True,
      {
        "IsMetric": False,
        "CustomLongitudinalEnabled": True,
        "CustomLongitudinalMode": "e2e",
        "ExperimentalMode": False,
        "DynamicExperimentalControl": True,
      },
      CardConfigSnapshot(False, True, False),
    ),
    (
      False,
      {
        "CustomLongitudinalEnabled": False,
        "ExperimentalMode": True,
        "DynamicExperimentalControl": True,
      },
      CardConfigSnapshot(False, False, True),
    ),
  ],
)
def test_read_card_config_snapshot(openpilot_longitudinal_control, values, expected):
  cp = car.CarParams(openpilotLongitudinalControl=openpilot_longitudinal_control)
  assert _read_card_config_snapshot(DummyParams(values), cp) == expected
