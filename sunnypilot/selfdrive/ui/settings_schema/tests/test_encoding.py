"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Proof that encode_numeric_option() reproduces the SCHEMA's numeric intent — the
single-source encoding that removes the torque-slider drift — and that the
enumerated-option detector correctly routes non-sequential enums to the
escape hatch.

Pure-logic: no pyray, runs headless.
"""
from __future__ import annotations

import pytest

from openpilot.sunnypilot.selfdrive.ui.settings_schema.encoding import (
  contiguous_int_options, encode_numeric_option, sequential_int_labels,
)
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import find_item, get_panel, load_schema

STEERING = get_panel(load_schema(), "steering")


def real_step(item):
  """Real-world step the encoded control moves in (value_change_step / scale)."""
  enc = encode_numeric_option(item)
  return enc.value_change_step / enc.scale


def test_lat_accel_factor_encoding_matches_schema_not_device():
  item = find_item(STEERING, "TorqueParamsOverrideLatAccelFactor")  # schema: 0.1 .. 5.0 step 0.1
  enc = encode_numeric_option(item)
  assert (enc.min_value, enc.max_value, enc.value_change_step, enc.use_float_scaling) == (10, 500, 10, True)
  assert enc.to_display(enc.min_value) == pytest.approx(0.1)
  assert enc.to_display(enc.max_value) == pytest.approx(5.0)
  # The whole point: the renderer steps in the schema's 0.1, not the device's hand-coded 0.01.
  assert real_step(item) == pytest.approx(0.1)
  assert real_step(item) != pytest.approx(0.01)


def test_friction_encoding_matches_schema():
  item = find_item(STEERING, "TorqueParamsOverrideFriction")  # schema: 0.0 .. 1.0 step 0.01
  enc = encode_numeric_option(item)
  assert (enc.min_value, enc.max_value, enc.value_change_step, enc.use_float_scaling) == (0, 100, 1, True)
  assert enc.to_display(enc.min_value) == pytest.approx(0.0)   # device hand-codes 0.01 here
  assert enc.to_display(enc.max_value) == pytest.approx(1.0)
  assert real_step(item) == pytest.approx(0.01)


def test_integer_option_passes_through_unscaled():
  item = find_item(STEERING, "BlinkerMinLateralControlSpeed")  # schema: 0 .. 255 step 5
  enc = encode_numeric_option(item)
  assert (enc.min_value, enc.max_value, enc.value_change_step, enc.use_float_scaling) == (0, 255, 5, False)
  assert enc.scale == 1
  assert real_step(item) == 5


def test_sequential_enum_yields_button_labels():
  item = find_item(STEERING, "MadsSteeringMode")  # values 0, 1, 2
  assert sequential_int_labels(item) == ["Remain Active", "Pause", "Disengage"]


def test_contiguous_int_enum_renders_as_labeled_stepper():
  # AutoLaneChangeTimer: -1..5 with a label per value. Matches the device's
  # option_item_sp stepper (lane_change_settings.py) — no value_map needed.
  enum = contiguous_int_options(find_item(STEERING, "AutoLaneChangeTimer"))
  assert enum is not None
  assert (enum.min_value, enum.max_value) == (-1, 5)
  assert enum.labels_by_value[-1] == "Off"
  assert enum.labels_by_value[0] == "Nudge"
  assert enum.labels_by_value[5] == "3 seconds"
  assert len(enum.labels_by_value) == 7


def test_zero_based_enum_is_also_contiguous():
  enum = contiguous_int_options(find_item(STEERING, "MadsSteeringMode"))  # 0,1,2
  assert enum is not None
  assert (enum.min_value, enum.max_value) == (0, 2)


@pytest.mark.parametrize("key", [
  "LiveTorqueSpeedAdaptiveMode",  # values off/shadow/apply (strings)
  "TorqueControlTune",            # values '', 1.0, 0.0 (string + floats)
])
def test_string_float_enums_remain_escape_hatch(key):
  # No contiguous-int and no zero-based-button-row encoding -> needs a value-mapped
  # or custom selector (deferred; these live in not-yet-wired sub-panels).
  item = find_item(STEERING, key)
  assert contiguous_int_options(item) is None
  assert sequential_int_labels(item) is None
