"""Tests for the 5.0 lateral demand stack abstraction.

Covers the LateralDemandStackId enum, the three concrete stack
classes, the resolver, the ControlsProfile auto-couple, and
the push_lateral_demand_stack_output contract surface used by
controlsd to forward stack_output.profile to LaC.
"""
import sys
import types

import pytest

params_pyx = types.ModuleType("openpilot.common.params_pyx")
params_pyx.Params = object
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

from openpilot.selfdrive.controls.lib.lateral_demand import ProcessedLateralDemand
from openpilot.sunnypilot.selfdrive.controls.lib.lateral_demand_stack import (
  CONTROLS_PROFILE_MAPPINGS,
  DEFAULT_CONTROLS_PROFILE,
  DEFAULT_LATERAL_DEMAND_STACK,
  ControlsProfileId,
  ControlsProfileMapping,
  CustomExperimentalLateralDemandStack,
  CustomV2LateralDemandStack,
  LateralDemandStackId,
  LateralDemandStackOutput,
  SunnypilotCurrentLateralDemandStack,
  controls_profile_id_for_name,
  controls_profile_mapping_for,
  lateral_demand_stack_id_for_name,
  resolve_lateral_demand_stack,
)


def _make_demand(processed_curvature: float = 0.001) -> ProcessedLateralDemand:
  return ProcessedLateralDemand(
    raw_curvature=processed_curvature,
    processed_curvature=processed_curvature,
    measured_curvature=0.0,
    curvature_limited=False,
    path_quality=1.0,
    path_reason="ok",
    lane_change_shaping_active=False,
    lane_change_blend=0.0,
    lateral_accel_limit=2.5,
    demand_source="model_path",
  )


# ---------------------------------------------------------------------------
# Resolver and registry
# ---------------------------------------------------------------------------


def test_lateral_demand_stack_id_for_name_resolves_known_values():
  assert lateral_demand_stack_id_for_name("sunnypilot-current") == LateralDemandStackId.SUNNYPILOT_CURRENT
  assert lateral_demand_stack_id_for_name("custom-2.0") == LateralDemandStackId.CUSTOM_V2
  assert lateral_demand_stack_id_for_name("custom-experimental") == LateralDemandStackId.CUSTOM_EXPERIMENTAL


def test_lateral_demand_stack_id_for_name_resolves_bytes():
  assert lateral_demand_stack_id_for_name(b"custom-experimental") == LateralDemandStackId.CUSTOM_EXPERIMENTAL


def test_lateral_demand_stack_id_for_name_returns_default_for_missing():
  assert lateral_demand_stack_id_for_name(None) == DEFAULT_LATERAL_DEMAND_STACK
  assert lateral_demand_stack_id_for_name(b"") == DEFAULT_LATERAL_DEMAND_STACK
  assert DEFAULT_LATERAL_DEMAND_STACK == LateralDemandStackId.SUNNYPILOT_CURRENT


def test_lateral_demand_stack_id_for_name_returns_default_for_unknown():
  assert lateral_demand_stack_id_for_name("not-a-stack") == DEFAULT_LATERAL_DEMAND_STACK
  assert lateral_demand_stack_id_for_name(b"\xff\xfe") == DEFAULT_LATERAL_DEMAND_STACK
  assert lateral_demand_stack_id_for_name(42) == DEFAULT_LATERAL_DEMAND_STACK


def test_resolve_lateral_demand_stack_returns_expected_class():
  assert isinstance(resolve_lateral_demand_stack("sunnypilot-current"), SunnypilotCurrentLateralDemandStack)
  assert isinstance(resolve_lateral_demand_stack("custom-2.0"), CustomV2LateralDemandStack)
  assert isinstance(resolve_lateral_demand_stack("custom-experimental"), CustomExperimentalLateralDemandStack)


def test_resolve_lateral_demand_stack_falls_back_to_default_for_unknown():
  stack = resolve_lateral_demand_stack("not-a-stack")
  assert isinstance(stack, SunnypilotCurrentLateralDemandStack)
  assert stack.stack_id == LateralDemandStackId.SUNNYPILOT_CURRENT


# ---------------------------------------------------------------------------
# Stack output contract
# ---------------------------------------------------------------------------


def test_lateral_demand_stack_update_returns_stack_output_with_legacy_and_profile():
  stack = resolve_lateral_demand_stack("sunnypilot-current")
  demand = _make_demand()
  out = stack.update(demand, v_ego=20.0)
  assert isinstance(out, LateralDemandStackOutput)
  assert out.legacy is demand
  assert out.profile.processed_curvature == pytest.approx(demand.processed_curvature)
  assert out.profile.mode
  assert out.profile.mode_confidence >= 0.0


def test_sunnypilot_current_stack_id():
  stack = SunnypilotCurrentLateralDemandStack()
  assert stack.stack_id == LateralDemandStackId.SUNNYPILOT_CURRENT


def test_custom_v2_stack_id():
  stack = CustomV2LateralDemandStack()
  assert stack.stack_id == LateralDemandStackId.CUSTOM_V2


def test_custom_experimental_stack_id():
  stack = CustomExperimentalLateralDemandStack()
  assert stack.stack_id == LateralDemandStackId.CUSTOM_EXPERIMENTAL


# ---------------------------------------------------------------------------
# ControlsProfile → LateralDemandStack + TorqueControlTune mapping
# ---------------------------------------------------------------------------


def test_controls_profile_default_is_standard():
  assert DEFAULT_CONTROLS_PROFILE == ControlsProfileId.STANDARD


def test_controls_profile_id_for_name_resolves_known_values():
  assert controls_profile_id_for_name("standard") == ControlsProfileId.STANDARD
  assert controls_profile_id_for_name("custom-2.0") == ControlsProfileId.CUSTOM_2
  assert controls_profile_id_for_name("experimental") == ControlsProfileId.EXPERIMENTAL


def test_controls_profile_id_for_name_returns_default_for_missing():
  assert controls_profile_id_for_name(None) == DEFAULT_CONTROLS_PROFILE
  assert controls_profile_id_for_name("not-a-profile") == DEFAULT_CONTROLS_PROFILE


def test_experimental_profile_maps_to_custom_experimental_and_torque_5_0():
  mapping = controls_profile_mapping_for(ControlsProfileId.EXPERIMENTAL)
  assert mapping.lateral_demand_stack == LateralDemandStackId.CUSTOM_EXPERIMENTAL
  assert mapping.torque_tune == 5.0


def test_custom_2_profile_maps_to_custom_v2_and_torque_4_1():
  mapping = controls_profile_mapping_for(ControlsProfileId.CUSTOM_2)
  assert mapping.lateral_demand_stack == LateralDemandStackId.CUSTOM_V2
  assert mapping.torque_tune == 4.1


def test_standard_profile_maps_to_sunnypilot_current_and_torque_4_1():
  mapping = controls_profile_mapping_for(ControlsProfileId.STANDARD)
  assert mapping.lateral_demand_stack == LateralDemandStackId.SUNNYPILOT_CURRENT
  assert mapping.torque_tune == 4.1


def test_all_mappings_cover_every_profile_id():
  seen = {m.profile_id for m in CONTROLS_PROFILE_MAPPINGS}
  assert seen == set(ControlsProfileId)
  for m in CONTROLS_PROFILE_MAPPINGS:
    assert isinstance(m, ControlsProfileMapping)
    assert m.lateral_demand_stack in set(LateralDemandStackId)
    assert m.torque_tune in (2.0, 2.1, 3.0, 4.0, 4.1, 5.0)
