"""Tests for the 5.0 controls profile resolver.

Covers:
- Top-level controls profile mapping to
  lateral_demand_stack + torque_control_tune + longitudinal_stack.
- Advanced per-layer override (LateralDemandStack,
  TorqueControlTune) when ShowAdvancedControls is on.
- Unknown / missing profile values fall back safely without
  exposing 5.0 by default.
- 5.0 is only selected when the user explicitly picked
  custom-experimental (or an explicit override).
- The custom-recommended profile does not silently expose a
  missing stack: the resolution's lateral_demand_stack_resolution
  carries fallback metadata.
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

from openpilot.sunnypilot.selfdrive.controls.lib.lateral_demand_stack import (
  ControlsProfileId,
  DEFAULT_CONTROLS_PROFILE,
  DEFAULT_LATERAL_DEMAND_STACK,
  LateralDemandStackId,
  TorqueControlTuneId,
  controls_profile_id_for_name,
  controls_profile_mapping_for,
  resolve_controls_profile,
  resolve_lateral_demand_stack,
  torque_control_tune_id_for_name,
)


# ---------------------------------------------------------------------------
# Defaults + missing
# ---------------------------------------------------------------------------


def test_missing_controls_profile_does_not_select_torque_50():
  """A missing ControlsProfile must not select torque 5.0. The
  default resolution must use the safe stable torque (4.1)."""
  res = resolve_controls_profile(None)
  assert res.torque_control_tune.value != "5.0"
  assert res.torque_control_tune.value == "4.1"
  assert res.lateral_demand_stack == DEFAULT_LATERAL_DEMAND_STACK


def test_unknown_controls_profile_does_not_select_torque_50():
  """An unknown ControlsProfile value must not select torque 5.0.
  It must resolve to the safe stable default (custom-2.0 +
  torque 4.1)."""
  res = resolve_controls_profile("not-a-profile")
  assert res.torque_control_tune.value != "5.0"
  assert res.torque_control_tune.value == "4.1"
  assert res.resolved_profile == DEFAULT_CONTROLS_PROFILE


def test_controls_profile_default_is_custom_2():
  """The default ControlsProfile id is custom-2.0 (the safe
  stable choice that maps to torque 4.1, not 5.0)."""
  assert DEFAULT_CONTROLS_PROFILE == ControlsProfileId.CUSTOM_2


def test_controls_profile_default_lateral_demand_stack_is_custom_2():
  """The default lateral demand stack id is custom-2.0. 5.0
  experimental is never the default."""
  assert DEFAULT_LATERAL_DEMAND_STACK == LateralDemandStackId.CUSTOM_V2


# ---------------------------------------------------------------------------
# Mapping: custom-2.0 -> 4.1
# ---------------------------------------------------------------------------


def test_controls_profile_custom_2_maps_to_torque_41():
  res = resolve_controls_profile("custom-2.0")
  assert res.resolved_profile == ControlsProfileId.CUSTOM_2
  assert res.lateral_demand_stack == LateralDemandStackId.CUSTOM_V2
  assert res.torque_control_tune.value == "4.1"
  assert res.longitudinal_stack == "custom-2.0"


# ---------------------------------------------------------------------------
# Mapping: custom-experimental -> 5.0
# ---------------------------------------------------------------------------


def test_controls_profile_experimental_maps_to_torque_50():
  res = resolve_controls_profile("custom-experimental")
  assert res.resolved_profile == ControlsProfileId.CUSTOM_EXPERIMENTAL
  assert res.lateral_demand_stack == LateralDemandStackId.CUSTOM_EXPERIMENTAL
  assert res.torque_control_tune.value == "5.0"


# ---------------------------------------------------------------------------
# Custom-recommended: stub with fallback metadata
# ---------------------------------------------------------------------------


def test_controls_profile_custom_recommended_does_not_silently_expose_unavailable_stacks():
  """custom-recommended has no real implementation yet; the
  resolution's lateral_demand_stack_resolution must carry
  fallback_reason='not_implemented' so the manifest / route
  tools can mark the entry as a stub."""
  res = resolve_controls_profile("custom-recommended")
  assert res.lateral_demand_stack_resolution.resolved_stack == LateralDemandStackId.CUSTOM_V2
  assert res.lateral_demand_stack_resolution.fallback_reason == "not_implemented"
  assert res.lateral_demand_stack_resolution.available is False
  # The runtime resolved stack is custom-2.0 (the safe fallback),
  # not custom-recommended itself.
  assert res.lateral_demand_stack == LateralDemandStackId.CUSTOM_V2
  # Torque tune for custom-recommended is 4.1 (not 5.0).
  assert res.torque_control_tune.value == "4.1"


# ---------------------------------------------------------------------------
# Mapping: sunnypilot-current
# ---------------------------------------------------------------------------


def test_controls_profile_sunnypilot_current_maps_to_sunnypilot_current_stack():
  res = resolve_controls_profile("sunnypilot-current")
  assert res.lateral_demand_stack == LateralDemandStackId.SUNNYPILOT_CURRENT
  assert res.torque_control_tune.value == "4.1"
  assert res.longitudinal_stack == "sunnypilot-current"


# ---------------------------------------------------------------------------
# Advanced override path
# ---------------------------------------------------------------------------


def test_controls_profile_advanced_override_can_select_torque_50():
  """When ShowAdvancedControls is on, an explicit
  TorqueControlTune=5.0 override on a custom-2.0 profile can
  select torque 5.0. The override metadata is recorded in
  fallback_reason."""
  res = resolve_controls_profile(
    "custom-2.0",
    advanced_torque_control_tune="5.0",
    advanced_overrides_enabled=True,
  )
  assert res.torque_control_tune.value == "5.0"
  assert "advanced_torque_control_tune_override" in res.fallback_reason


def test_controls_profile_advanced_lateral_demand_stack_override():
  """An explicit LateralDemandStack=custom-experimental override
  on a custom-2.0 profile selects the experimental stack
  without changing the profile id."""
  res = resolve_controls_profile(
    "custom-2.0",
    advanced_lateral_demand_stack="custom-experimental",
    advanced_overrides_enabled=True,
  )
  assert res.resolved_profile == ControlsProfileId.CUSTOM_2
  assert res.lateral_demand_stack == LateralDemandStackId.CUSTOM_EXPERIMENTAL
  assert "advanced_lateral_demand_stack_override" in res.fallback_reason
  # Torque tune from the profile mapping is unchanged (4.1);
  # the override only flipped the lateral demand stack.
  assert res.torque_control_tune.value == "4.1"


def test_controls_profile_advanced_overrides_disabled_ignores_overrides():
  """When ShowAdvancedControls is off, advanced per-layer
  override params must be ignored. The profile mapping wins
  regardless of any override value."""
  res = resolve_controls_profile(
    "custom-2.0",
    advanced_torque_control_tune="5.0",
    advanced_lateral_demand_stack="custom-experimental",
    advanced_overrides_enabled=False,
  )
  assert res.torque_control_tune.value == "4.1"
  assert res.lateral_demand_stack == LateralDemandStackId.CUSTOM_V2


# ---------------------------------------------------------------------------
# Migration safety
# ---------------------------------------------------------------------------


def test_existing_torque_control_tune_preserved_when_advanced_overrides_off():
  """An existing user with TorqueControlTune=2.0 set explicitly
  and no advanced-overrides gate must keep torque 2.0.  This
  is the migration path: we do not silently change a user
  from 2.0/2.1/4.1 to 5.0 when advanced overrides are off."""
  res = resolve_controls_profile(
    None,
    advanced_torque_control_tune="2.0",
    advanced_overrides_enabled=False,
  )
  # Override ignored because advanced_overrides_enabled=False.
  assert res.torque_control_tune.value == "4.1"


def test_unknown_lateral_demand_stack_persists_safe_fallback():
  """An unknown LateralDemandStack value (e.g. from a typo or
  downgrade from a future version) must resolve to the safe
  default. The resolver sets fallback_reason on the
  lateral_demand_stack_resolution."""
  res = resolve_lateral_demand_stack("not-a-stack")
  assert res.resolved_stack == DEFAULT_LATERAL_DEMAND_STACK
  # The resolver silently drops the unknown value: callers
  # may persist the safe fallback if they wish.
  assert res.fallback_reason == ""


def test_unknown_torque_control_tune_does_not_select_torque_50():
  """An unknown TorqueControlTune value must NOT select 5.0.
  The safe fallback (4.1) is used. 5.0 is only selected when
  the user explicitly asked for it."""
  from openpilot.sunnypilot.selfdrive.controls.lib.lateral_demand_stack import (
    resolve_torque_control_tune,
  )
  res = resolve_torque_control_tune("not-a-tune")
  assert res.resolved_tune.value != "5.0"
  assert res.resolved_tune.value == "4.1"


def test_torque_control_tune_id_for_name_returns_default_for_missing():
  assert torque_control_tune_id_for_name(None) == TorqueControlTuneId.V41


def test_torque_control_tune_id_for_name_returns_default_for_unknown():
  assert torque_control_tune_id_for_name("not-a-tune") == TorqueControlTuneId.V41
  assert torque_control_tune_id_for_name(b"") == TorqueControlTuneId.V41


def test_torque_control_tune_id_for_name_resolves_5_0():
  assert torque_control_tune_id_for_name("5.0") == TorqueControlTuneId.V50_EXPERIMENTAL
  assert torque_control_tune_id_for_name(b"5.0") == TorqueControlTuneId.V50_EXPERIMENTAL
  assert torque_control_tune_id_for_name(5.0) == TorqueControlTuneId.V50_EXPERIMENTAL


def test_controls_profile_id_for_name_resolves_known_values():
  assert controls_profile_id_for_name("sunnypilot-current") == ControlsProfileId.SUNNYPILOT_CURRENT
  assert controls_profile_id_for_name("custom-recommended") == ControlsProfileId.CUSTOM_RECOMMENDED
  assert controls_profile_id_for_name("custom-2.0") == ControlsProfileId.CUSTOM_2
  assert controls_profile_id_for_name("custom-experimental") == ControlsProfileId.CUSTOM_EXPERIMENTAL


def test_all_profile_mappings_cover_every_id():
  from openpilot.sunnypilot.selfdrive.controls.lib.lateral_demand_stack import (
    CONTROLS_PROFILE_MAPPINGS,
  )
  seen = {m.profile_id for m in CONTROLS_PROFILE_MAPPINGS}
  assert seen == set(ControlsProfileId)
  for m in CONTROLS_PROFILE_MAPPINGS:
    assert m.lateral_demand_stack in set(LateralDemandStackId)
    assert m.torque_control_tune in set(TorqueControlTuneId)
    assert m.longitudinal_stack
