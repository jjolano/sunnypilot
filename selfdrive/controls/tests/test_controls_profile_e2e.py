import openpilot.selfdrive.controls.lib.feature_registry_entries  # noqa: F401  (registers entries on import)

from openpilot.selfdrive.controls.lib.controls_profile_resolver import (
  ControlsProfileResolver,
  LATERAL_STACK_PARAM,
  LONGITUDINAL_STACK_PARAM,
)
from openpilot.selfdrive.controls.lib.feature_registry import get_features
from openpilot.selfdrive.controls.lib.lateral_demand_stacks import (
  CUSTOM_V2 as LATERAL_CUSTOM_V2,
  resolve_lateral_demand_stack,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks import (
  CUSTOM_V2 as LONG_CUSTOM_V2,
  make_custom_longitudinal_stack,
  resolve_longitudinal_stack,
)
from openpilot.selfdrive.controls.lib.policy_facade import resolve_controls_profile
from openpilot.selfdrive.controls.lib.ui_metadata import get_controls_profile_metadata


class _FakeParams:
  def __init__(self, store: dict[str, bytes]) -> None:
    self._store = dict(store)

  def __call__(self, key: str, default: str) -> str:
    raw = self._store.get(key)
    if raw is None:
      return default
    if isinstance(raw, bytes):
      return raw.decode("utf-8")
    return str(raw)


class _FakeCP:
  brand = "HONDA"
  carFingerprint = "HONDA_CIVIC"
  openpilotLongitudinalControl = True
  alphaLongitudinalAvailable = True
  pcmCruise = True
  radarUnavailable = False


def test_end_to_end_default_pair():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_FakeCP())
  assert profile.resolved_lateral == LATERAL_CUSTOM_V2
  assert profile.resolved_longitudinal == LONG_CUSTOM_V2
  assert profile.lateral_features.features.one_pedal_longitudinal is True
  assert profile.longitudinal_features.features.one_pedal_longitudinal is True
  assert profile.lateral_features.features.experimental_stage == "stable"
  assert profile.is_custom is True


def test_end_to_end_experimental_pair():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-experimental", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_FakeCP())
  assert profile.resolved_lateral == "custom-experimental"
  assert profile.resolved_longitudinal == "custom-2.0"
  assert profile.lateral_features.features.experimental_stage == "experimental"
  assert profile.longitudinal_features.features.experimental_stage == "stable"


def test_end_to_end_baseline_pair():
  params = _FakeParams({LATERAL_STACK_PARAM: b"sunnypilot-current", LONGITUDINAL_STACK_PARAM: b"sunnypilot-current"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_FakeCP())
  assert profile.is_pure_baseline is True
  assert profile.lateral_features.features.one_pedal_longitudinal is False


def test_end_to_end_resolver_to_factory_to_stack():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_FakeCP())

  long_stack = make_custom_longitudinal_stack(profile.resolved_longitudinal)
  assert type(long_stack).__name__ in ("CustomLongitudinalStackV2", "CustomExperimentalLongitudinalStack")

  lateral_resolution = resolve_lateral_demand_stack(profile.resolved_lateral)
  assert lateral_resolution.resolved_stack == LATERAL_CUSTOM_V2


def test_end_to_end_ui_metadata_matches_resolved_features():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_FakeCP())
  md = get_controls_profile_metadata()
  v2_features = md["feature_registry"]["custom-2.0"]["features"]
  assert profile.lateral_features.features.one_pedal_longitudinal == v2_features["one_pedal_longitudinal"]
  assert profile.lateral_features.features.experimental_stage == v2_features["experimental_stage"]


def test_end_to_end_manifests_agree_with_selector():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-experimental", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_FakeCP())
  assert "custom-experimental" in get_controls_profile_metadata()["lateral_manifest"]["stacks"]
  assert "custom-experimental" not in get_controls_profile_metadata()["longitudinal_manifest"]["stacks"]
  assert get_features("custom-experimental").experimental_stage == "experimental"


def test_end_to_end_fallback_path():
  class _IncapableCP:
    brand = "TOYOTA"
    carFingerprint = "TOYOTA_COROLLA"
    openpilotLongitudinalControl = False
    alphaLongitudinalAvailable = False
    pcmCruise = True
    radarUnavailable = False
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_IncapableCP())
  assert profile.resolved_lateral == LATERAL_CUSTOM_V2
  assert profile.resolved_longitudinal == "sunnypilot-current"
  assert profile.longitudinal_fallback_active is True
  assert profile.longitudinal_fallback_reason == "unavailable_stack"
  assert profile.lateral_features.features.one_pedal_longitudinal is True
  assert profile.longitudinal_features.features.one_pedal_longitudinal is False
