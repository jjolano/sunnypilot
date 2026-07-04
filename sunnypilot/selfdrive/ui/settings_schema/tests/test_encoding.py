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
  contiguous_int_options, encode_numeric_option, homogeneous_string_options, sequential_int_labels, string_option_index, value_mapped_option,
)
from openpilot.sunnypilot.selfdrive.ui.settings_schema.schema_loader import find_item, get_panel, load_schema

STEERING = get_panel(load_schema(), "steering")
CRUISE = get_panel(load_schema(), "cruise")


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


def test_homogeneous_string_enum_yields_mapped_button_row():
  item = find_item(CRUISE, "CustomLongitudinalMode")  # values acc/e2e/scc (strings)
  enum = homogeneous_string_options(item)
  assert enum is not None
  assert enum.values == ["acc", "e2e", "scc"]
  assert enum.labels == ["ACC", "E2E", "SCC"]

  lead_enum = homogeneous_string_options(find_item(CRUISE, "LeadAnticipationMode"))
  assert lead_enum is not None
  assert lead_enum.values == ["off", "shadow", "apply"]
  assert lead_enum.labels == ["Off", "Monitor only", "Apply lead smoothing"]

  follow_gap_enum = homogeneous_string_options(find_item(CRUISE, "DynamicFollowGapMode"))
  assert follow_gap_enum is not None
  assert follow_gap_enum.values == ["off", "shadow", "apply"]
  assert follow_gap_enum.labels == ["Off", "Monitor only", "Apply follow gap"]

  debug_enum = homogeneous_string_options(find_item(CRUISE, "LongitudinalDebugTraceMode"))
  assert debug_enum is not None
  assert debug_enum.values == ["off", "log"]
  assert debug_enum.labels == ["Off", "Log"]

  roll_enum = homogeneous_string_options(find_item(STEERING, "RollCompGainMode"))
  assert roll_enum is not None
  assert roll_enum.values == ["off", "shadow", "apply"]
  assert roll_enum.labels == ["Off", "Learn only", "Apply learned gain"]

  lane_rate_enum = homogeneous_string_options(find_item(STEERING, "LaneRateDampingMode"))
  assert lane_rate_enum is not None
  assert lane_rate_enum.values == ["off", "shadow", "apply"]
  assert lane_rate_enum.labels == ["Off", "Monitor only", "Apply"]

  cut_in_enum = homogeneous_string_options(find_item(CRUISE, "CutInBrakeAssistMode"))
  assert cut_in_enum is not None
  assert cut_in_enum.values == ["off", "shadow", "apply"]
  assert cut_in_enum.labels == ["Off", "Monitor only", "Apply gentle cap"]

  curve_enum = homogeneous_string_options(find_item(CRUISE, "CurveSpeedConfidenceMode"))
  assert curve_enum is not None
  assert curve_enum.values == ["off", "shadow", "apply_conservative"]
  assert curve_enum.labels == ["Off", "Monitor only", "Apply conservative"]

  curve_traffic_enum = homogeneous_string_options(find_item(CRUISE, "CurveTrafficAdvisorMode"))
  assert curve_traffic_enum is not None
  assert curve_traffic_enum.values == ["off", "shadow", "apply_conservative"]
  assert curve_traffic_enum.labels == ["Off", "Monitor only", "Apply conservative"]

  release_enum = homogeneous_string_options(find_item(CRUISE, "StandstillReleaseConfidenceMode"))
  assert release_enum is not None
  assert release_enum.values == ["off", "shadow", "gate"]
  assert release_enum.labels == ["Off", "Monitor only", "Release gate"]


def test_custom_longitudinal_string_index_matches_planner_fallbacks():
  enum = homogeneous_string_options(find_item(CRUISE, "CustomLongitudinalMode"))
  assert enum is not None
  assert string_option_index("", enum, "CustomLongitudinalMode") == 2       # missing/empty -> default SCC
  assert string_option_index("0", enum, "CustomLongitudinalMode") == 0      # legacy ACC
  assert string_option_index(b"1", enum, "CustomLongitudinalMode") == 1     # legacy E2E bytes-safe
  assert string_option_index("bad", enum, "CustomLongitudinalMode") == 0    # invalid -> planner ACC fallback


def test_longitudinal_debug_trace_string_index_defaults_to_off():
  enum = homogeneous_string_options(find_item(CRUISE, "LongitudinalDebugTraceMode"))
  assert enum is not None
  assert string_option_index("", enum, "LongitudinalDebugTraceMode") == 0
  assert string_option_index("off", enum, "LongitudinalDebugTraceMode") == 0
  assert string_option_index("log", enum, "LongitudinalDebugTraceMode") == 1
  assert string_option_index("bad", enum, "LongitudinalDebugTraceMode") == 0


def test_roll_comp_gain_string_index_defaults_to_off():
  enum = homogeneous_string_options(find_item(STEERING, "RollCompGainMode"))
  assert enum is not None
  assert string_option_index("", enum, "RollCompGainMode") == 0
  assert string_option_index("off", enum, "RollCompGainMode") == 0
  assert string_option_index("shadow", enum, "RollCompGainMode") == 1
  assert string_option_index("apply", enum, "RollCompGainMode") == 2
  assert string_option_index("bad", enum, "RollCompGainMode") == 0


def test_lane_rate_damping_string_index_defaults_to_off():
  enum = homogeneous_string_options(find_item(STEERING, "LaneRateDampingMode"))
  assert enum is not None
  assert string_option_index("", enum, "LaneRateDampingMode") == 0
  assert string_option_index("off", enum, "LaneRateDampingMode") == 0
  assert string_option_index("shadow", enum, "LaneRateDampingMode") == 1
  assert string_option_index("apply", enum, "LaneRateDampingMode") == 2
  assert string_option_index("bad", enum, "LaneRateDampingMode") == 0


def test_shadow_observability_string_indexes_default_to_off():
  for key in ("CutInBrakeAssistMode", "CurveSpeedConfidenceMode", "CurveTrafficAdvisorMode", "StandstillReleaseConfidenceMode"):
    enum = homogeneous_string_options(find_item(CRUISE, key))
    assert enum is not None
    assert string_option_index("", enum, key) == 0
    assert string_option_index("off", enum, key) == 0
    assert string_option_index("shadow", enum, key) == 1
    assert string_option_index("bad", enum, key) == 0
  assert string_option_index("apply", homogeneous_string_options(find_item(CRUISE, "CutInBrakeAssistMode")), "CutInBrakeAssistMode") == 2
  assert string_option_index("apply_conservative", homogeneous_string_options(find_item(CRUISE, "CurveSpeedConfidenceMode")), "CurveSpeedConfidenceMode") == 2
  assert string_option_index("apply_conservative", homogeneous_string_options(find_item(CRUISE, "CurveTrafficAdvisorMode")), "CurveTrafficAdvisorMode") == 2
  assert string_option_index("gate", homogeneous_string_options(find_item(CRUISE, "StandstillReleaseConfidenceMode")), "StandstillReleaseConfidenceMode") == 2


def test_mixed_string_float_enum_remains_escape_hatch():
  key = "TorqueControlTune"  # values '', 1.0, 0.0 (string + floats)
  item = find_item(STEERING, key)
  assert contiguous_int_options(item) is None
  assert sequential_int_labels(item) is None
  assert value_mapped_option(item) is None  # value_mapped only handles ints
  assert homogeneous_string_options(item) is None


def test_value_mapped_option_for_gapped_ints():
  # Cruise's CustomAccLongPressIncrement: stored values 1/5/10 over a 3-step stepper.
  item = {"options": [{"value": 1, "label": "1"}, {"value": 5, "label": "5"}, {"value": 10, "label": "10"}]}
  vm = value_mapped_option(item)
  assert vm is not None
  assert vm.value_map == {1: 1, 2: 5, 3: 10}           # display index -> stored value
  assert vm.labels_by_value == {1: "1", 5: "5", 10: "10"}
  assert (vm.min_value, vm.max_value) == (1, 3)


def test_value_mapped_option_rejects_non_ints():
  assert value_mapped_option({"options": [{"value": "a", "label": "A"}]}) is None
  assert value_mapped_option({"options": []}) is None
