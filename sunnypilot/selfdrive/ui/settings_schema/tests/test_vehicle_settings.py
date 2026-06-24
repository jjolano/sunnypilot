"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Parity for per-brand vehicle settings migrated to the schema-driven renderer.
Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import load_schema

VEHICLE_SETTINGS = load_schema().get("vehicle_settings", {})

VEHICLE_EXPECTED = {
  "hyundai": {
    "HyundaiLongitudinalTuning": "multiple_button",
  },
  "subaru": {
    "SubaruStopAndGo": "toggle",
    "SubaruStopAndGoManualParkingBrake": "toggle",
  },
  "tesla": {
    "TeslaCoopSteering": "toggle",
  },
  "toyota": {
    "ToyotaEnforceStockLongitudinal": "toggle",
    "ToyotaTSS2SmoothLongitudinal": "toggle",
    "ToyotaStopAndGoHack": "toggle",
  },
}


def _items_iter(brand_data: dict):
  """Yield top-level items from a brand's vehicle settings."""
  for item in brand_data.get("items", []):
    yield item
  for section in brand_data.get("sections", []):
    for item in section.get("items", []):
      yield item


def test_vehicle_settings_brands():
  assert set(VEHICLE_SETTINGS.keys()) == set(VEHICLE_EXPECTED.keys())


def test_vehicle_settings_parity_exact_keys_and_widgets():
  for brand, expected in VEHICLE_EXPECTED.items():
    brand_data = VEHICLE_SETTINGS.get(brand, {})
    got = {it["key"]: it["widget"] for it in _items_iter(brand_data) if "key" in it}
    assert got == expected, f"{brand} mismatch: {got} != {expected}"


def test_vehicle_settings_toyota_needs_onroad_cycle():
  brand_data = VEHICLE_SETTINGS.get("toyota", {})
  for item in _items_iter(brand_data):
    if item.get("key") in ("ToyotaEnforceStockLongitudinal", "ToyotaTSS2SmoothLongitudinal", "ToyotaStopAndGoHack"):
      assert item.get("needs_onroad_cycle") is True, f"{item['key']} must declare needs_onroad_cycle"
