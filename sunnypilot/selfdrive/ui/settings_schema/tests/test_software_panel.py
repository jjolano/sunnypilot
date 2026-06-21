"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Parity for the software settings panel migrated to the schema-driven renderer.
Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, iter_items, load_schema

SOFTWARE = get_panel(load_schema(), "software")

SOFTWARE_EXPECTED = {
  "OnroadUpdateNotice": "info",
  "UpdaterCurrentDescription": "info",
  "DownloadUpdate": "button",
  "InstallUpdate": "button",
  "UpdaterTargetBranch": "button",
  "Uninstall": "button",
  "DisableUpdates": "toggle",
}


def test_software_parity_exact_keys_and_widgets():
  got = {it["key"]: it["widget"] for it in iter_items(SOFTWARE) if "key" in it}
  assert got == SOFTWARE_EXPECTED
