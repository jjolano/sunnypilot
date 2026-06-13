"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The consolidated Interface settings page: Visuals (On-Road Display) + Display
(Screen) folded into one panel under two group headers. Both sources are flat
(no sub-panels), so the layout has no nav buttons.

`interface_panel_dict` is pure (headless-testable); `build_interface_layout`
wraps it in a SchemaNavLayout, reusing its section-header rendering.
"""
from openpilot.sunnypilot.selfdrive.ui.settings_schema.consolidation import combined_panel


def interface_panel_dict(schema: dict | None = None) -> dict:
  return combined_panel("interface", "Interface",
                        [("On-Road Display", "visuals"), ("Screen", "display")], schema)


def build_interface_layout():
  from openpilot.sunnypilot.selfdrive.ui.settings_schema.nav_layout import SchemaNavLayout
  return SchemaNavLayout(interface_panel_dict(), {})
