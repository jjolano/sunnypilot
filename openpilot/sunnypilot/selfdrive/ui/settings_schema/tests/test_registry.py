"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Tests for the escape-hatch registries (option providers + custom widgets). Pure,
headless. The headline check: the device's torque-versions provider produces the
exact option list the cloud schema injects, so the dynamic options can't diverge.
"""
from __future__ import annotations

from openpilot.sunnypilot.selfdrive.ui.settings_schema.registry import (
  custom_widget_factory, register_custom_widget, register_option_provider,
  resolve_options, torque_version_options,
)


def test_literal_options_take_precedence():
  opts = [{"value": 0, "label": "A"}]
  assert resolve_options({"options": opts}) is opts


def test_options_source_resolves_via_provider():
  register_option_provider("unit_test_src", lambda: [{"value": 1, "label": "X"}])
  assert resolve_options({"options_source": "unit_test_src"}) == [{"value": 1, "label": "X"}]


def test_unregistered_source_yields_none():
  assert resolve_options({"options_source": "does_not_exist"}) is None


def test_no_options_yields_none():
  assert resolve_options({"widget": "toggle"}) is None


def test_torque_versions_provider_registered_with_default_first():
  opts = resolve_options({"options_source": "torque_versions"})
  assert opts is not None
  assert opts[0] == {"value": "", "label": "Default"}


def test_torque_provider_matches_cloud_injection():
  # The device provider must produce the exact list the cloud schema injects
  # (generate_settings_schema._build_torque_options) — same dynamic options on both sides.
  from openpilot.sunnypilot.sunnylink.tools import generate_settings_schema as gss
  expected = gss._build_torque_options(gss._load_torque_versions())
  assert torque_version_options() == expected


def test_custom_widget_register_and_lookup():
  sentinel = object()
  register_custom_widget("unit_test_widget", lambda item: sentinel)
  factory = custom_widget_factory("unit_test_widget")
  assert factory is not None
  assert factory({"key": "X"}) is sentinel


def test_missing_custom_widget_is_none():
  assert custom_widget_factory("nope_not_here") is None
