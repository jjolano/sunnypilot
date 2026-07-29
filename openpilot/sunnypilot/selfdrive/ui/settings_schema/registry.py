"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Escape-hatch registries: named option providers + custom-widget components.

Two declarative deferral points the schema can use when a control is more than a
plain toggle/option/multi-button:

  - `options_source: <name>`        — a control's option list comes from a device
                                       provider (dynamic data), not literal `options`.
  - `widget: custom, component: <id>` — the control is built by a device-registered
                                       factory (dialog pickers, bespoke widgets).

This mirrors the cloud's dynamic-option injection
(`generate_settings_schema._inject_dynamic_options`) so device and cloud render
the same dynamic content. Pure module — no pyray; the registered custom-widget
factories carry the rendering deps, the registry itself does not.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable

from openpilot.common.basedir import BASEDIR

OptionProvider = Callable[[], list[dict]]
CustomWidgetFactory = Callable[[dict], object]

_OPTION_PROVIDERS: dict[str, OptionProvider] = {}
_CUSTOM_WIDGETS: dict[str, CustomWidgetFactory] = {}


# --- option providers --------------------------------------------------------

def register_option_provider(name: str, provider: OptionProvider) -> None:
  _OPTION_PROVIDERS[name] = provider


def resolve_options(item: dict) -> list[dict] | None:
  """Return a control's options: literal `options` win, else `options_source`.

  None means "no enumerated options" (a numeric option or a non-enumerated widget).
  An `options_source` naming an unregistered provider also yields None — the caller
  routes that to the escape hatch rather than rendering an empty selector.
  """
  if "options" in item:
    return item["options"]
  source = item.get("options_source")
  if source is not None:
    provider = _OPTION_PROVIDERS.get(source)
    if provider is not None:
      return provider()
  return None


# --- custom widgets ----------------------------------------------------------

def register_custom_widget(component: str, factory: CustomWidgetFactory) -> None:
  _CUSTOM_WIDGETS[component] = factory


def custom_widget_factory(component: str) -> CustomWidgetFactory | None:
  return _CUSTOM_WIDGETS.get(component)


# --- built-in providers ------------------------------------------------------

TORQUE_VERSIONS_PATH = os.path.join(
  BASEDIR, "openpilot", "sunnypilot", "selfdrive", "controls", "lib", "latcontrol_torque_versions.json"
)


def torque_version_options() -> list[dict]:
  """Options for TorqueControlTune, sourced from the versions manifest.

  Byte-for-byte the same list the cloud injects
  (generate_settings_schema._build_torque_options): a leading Default plus each
  version, newest first. Keeping one shape on both sides is the whole point of the
  named-provider hatch.
  """
  options: list[dict] = [{"value": "", "label": "Default"}]
  try:
    with open(TORQUE_VERSIONS_PATH) as f:
      versions = json.load(f)
  except (FileNotFoundError, json.JSONDecodeError):
    return options

  parsed: list[tuple[float, str]] = []
  for label, info in versions.items():
    try:
      parsed.append((float(info["version"]), label))
    except (KeyError, TypeError, ValueError):
      continue
  for version, label in sorted(parsed, key=lambda kv: kv[0], reverse=True):
    options.append({"value": version, "label": label})
  return options


register_option_provider("torque_versions", torque_version_options)
