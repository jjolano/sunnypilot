"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Parity for the developer settings panel migrated to the schema-driven renderer.
Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, iter_items, load_schema

DEVELOPER = get_panel(load_schema(), "developer")

DEVELOPER_EXPECTED = {
  "AdbEnabled": "toggle",
  "SshEnabled": "toggle",
  "JoystickDebugMode": "toggle",
  "AlphaLongitudinalEnabled": "toggle",
  "ShowDebugInfo": "toggle",
  "LateralManeuverMode": "toggle",
  "LongitudinalManeuverMode": "toggle",
  "ShowAdvancedControls": "toggle",
  "EnableGithubRunner": "toggle",
  "EnableCopyparty": "toggle",
  "QuickBootToggle": "toggle",
  "ErrorLog": "button",
  "TailscaleInstall": "button",
  "EnableTailscale": "toggle",
  "TailscaleLogin": "button",
  "TailscaleLogout": "button",
}


def test_developer_parity_exact_keys_and_widgets():
  got = {it["key"]: it["widget"] for it in iter_items(DEVELOPER) if "key" in it}
  assert got == DEVELOPER_EXPECTED


def test_developer_alpha_longitudinal_needs_onroad_cycle():
  item = next(it for it in iter_items(DEVELOPER) if it.get("key") == "AlphaLongitudinalEnabled")
  assert item.get("needs_onroad_cycle") is True
