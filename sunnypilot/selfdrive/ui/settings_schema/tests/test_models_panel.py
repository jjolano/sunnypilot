"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Parity for the models settings panel migrated to the schema-driven renderer.
Pure/headless.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import get_panel, iter_items, load_schema

MODELS = get_panel(load_schema(), "models")

MODELS_EXPECTED = {
  "CurrentModel": "button",
  "CancelDownload": "button",
  "ModelProgress": "custom",
  "RefreshModelList": "button",
  "ClearModelCache": "button",
  "LaneTurnDesire": "toggle",
  "LaneTurnValue": "option",
  "LagdToggle": "toggle",
  "LagdToggleDelay": "option",
  "CameraOffset": "option",
}


def test_models_parity_exact_keys_and_widgets():
  got = {it["key"]: it["widget"] for it in iter_items(MODELS) if "key" in it}
  assert got == MODELS_EXPECTED
