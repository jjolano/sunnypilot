import openpilot.selfdrive.controls.lib.feature_registry_entries  # noqa: F401  (registers entries)
from openpilot.selfdrive.controls.lib.policy_facade import (
  ControlsProfile,
  policy_summary,
  resolve_controls_profile,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.selector import (
  CUSTOM_V2 as LATERAL_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LATERAL_SUNNYPILOT_CURRENT,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import (
  CUSTOM_V2 as LONG_CUSTOM_V2,
  SUNNYPILOT_CURRENT as LONG_SUNNYPILOT_CURRENT,
)


class _FakeCP:
  brand = "HONDA"
  carFingerprint = "HONDA_CIVIC"
  openpilotLongitudinalControl = True
  alphaLongitudinalAvailable = True
  pcmCruise = True
  radarUnavailable = False


class _IncapableCP:
  brand = "TOYOTA"
  carFingerprint = "TOYOTA_COROLLA"
  openpilotLongitudinalControl = False
  alphaLongitudinalAvailable = False
  pcmCruise = True
  radarUnavailable = False


def test_resolve_profile_happy_path():
  profile = resolve_controls_profile(LATERAL_CUSTOM_V2, LONG_CUSTOM_V2, CP=_FakeCP())
  assert isinstance(profile, ControlsProfile)
  assert profile.requested_lateral == LATERAL_CUSTOM_V2
  assert profile.resolved_lateral == LATERAL_CUSTOM_V2
  assert profile.requested_longitudinal == LONG_CUSTOM_V2
  assert profile.resolved_longitudinal == LONG_CUSTOM_V2
  assert profile.lateral_fallback_active is False
  assert profile.longitudinal_fallback_active is False
  assert profile.is_custom is True
  assert profile.is_pure_baseline is False


def test_resolve_profile_baseline_pair_is_pure_baseline():
  profile = resolve_controls_profile(LATERAL_SUNNYPILOT_CURRENT, LONG_SUNNYPILOT_CURRENT, CP=_FakeCP())
  assert profile.is_pure_baseline is True
  assert profile.is_custom is False


def test_resolve_profile_longitudinal_fallback_on_incapable_car():
  profile = resolve_controls_profile(LATERAL_CUSTOM_V2, LONG_CUSTOM_V2, CP=_IncapableCP())
  assert profile.resolved_longitudinal == LONG_SUNNYPILOT_CURRENT
  assert profile.longitudinal_fallback_active is True
  assert profile.longitudinal_fallback_reason == "unavailable_stack"
  assert profile.is_custom is True


def test_resolve_profile_features_match_resolved_stack():
  profile = resolve_controls_profile("custom-2.0", "custom-2.0", CP=_FakeCP())
  assert profile.lateral_features.features.one_pedal_longitudinal is True
  assert profile.longitudinal_features.features.one_pedal_longitudinal is True
  assert profile.lateral_features.features.experimental_stage == "stable"


def test_resolve_profile_experimental_stage_propagates():
  profile = resolve_controls_profile("custom-experimental", "custom-experimental", CP=_FakeCP())
  assert profile.lateral_features.features.experimental_stage == "experimental"
  assert profile.longitudinal_features.features.experimental_stage == "experimental"


def test_policy_summary_round_trip():
  profile = resolve_controls_profile("custom-2.0", "custom-2.0", CP=_FakeCP())
  summary = policy_summary(profile)
  assert summary["lateral"]["resolved"] == "custom-2.0"
  assert summary["longitudinal"]["resolved"] == "custom-2.0"
  assert summary["is_custom"] is True
  assert summary["is_pure_baseline"] is False
  assert summary["lateral"]["features"]["one_pedal_longitudinal"] is True
