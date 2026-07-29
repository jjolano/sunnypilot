"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The consolidated Driving settings page.

Combines the steering (Lateral) and cruise (Longitudinal) schema panels into one
page under two group headers, with the sub-section titles suppressed (so controls
stay interleaved) and the redundant "Enable " prefix dropped from toggle titles.
This is a device-side runtime combine — the steering/cruise schema sources and the
cloud frontend are untouched; only the device IA consolidates.

`driving_panel_dict` is pure (no pyray) so it can be unit-tested headless;
`build_driving_layout` wraps it in a SchemaNavLayout with the hand-coded sub-layouts.
"""
from openpilot.sunnypilot.selfdrive.ui.settings_schema.consolidation import combined_panel


def driving_panel_dict(schema: dict | None = None) -> dict:
  return combined_panel("driving", "Driving",
                        [("Lateral Control", "steering"), ("Longitudinal Control", "cruise")], schema)


def build_driving_layout():
  """Construct the live Driving panel (lazy pyray imports)."""
  from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings import (
    SpeedLimitSettingsLayout,
  )
  from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.mads_settings import MadsSettingsLayout
  from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.torque_settings import (
    TorqueSettingsLayout,
  )
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.nav_layout import SchemaNavLayout
  return SchemaNavLayout(driving_panel_dict(), {
    "mads_settings": MadsSettingsLayout,
    "torque_settings": TorqueSettingsLayout,
    "speed_limit_settings": SpeedLimitSettingsLayout,
  })
