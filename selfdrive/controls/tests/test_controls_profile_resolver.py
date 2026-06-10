import openpilot.selfdrive.controls.lib.feature_registry_entries  # noqa: F401
from openpilot.selfdrive.controls.lib.controls_profile_resolver import (
  ControlsProfileResolver,
  LATERAL_STACK_PARAM,
  LONGITUDINAL_STACK_PARAM,
  active_profile_summary,
  make_default_resolver,
)


class _FakeParams:
  def __init__(self, store: dict[str, bytes | str]) -> None:
    self._store = {k: (v.encode() if isinstance(v, str) else v) for k, v in store.items()}

  def __call__(self, key: str, default: str) -> str:
    raw = self._store.get(key, default)
    if isinstance(raw, bytes):
      return raw.decode("utf-8")
    return raw


class _FakeCP:
  brand = "HONDA"
  carFingerprint = "HONDA_CIVIC"
  openpilotLongitudinalControl = True
  alphaLongitudinalAvailable = True
  pcmCruise = True
  radarUnavailable = False


def test_resolver_returns_profile_from_params():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.resolve(CP=_FakeCP())
  assert profile.resolved_lateral == "custom-2.0"
  assert profile.resolved_longitudinal == "custom-2.0"


def test_resolver_engage_latches_profile():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  profile = resolver.engage(CP=_FakeCP())
  assert profile.resolved_lateral == "custom-2.0"
  assert resolver.latch_active is True
  assert resolver.engaged is True


def test_latch_holds_against_param_change():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  resolver.engage(CP=_FakeCP())
  params._store[LATERAL_STACK_PARAM] = b"sunnypilot-current"
  current = resolver.current_profile(CP=_FakeCP())
  assert current is not None
  assert current.resolved_lateral == "custom-2.0"


def test_release_clears_latch():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  resolver.engage(CP=_FakeCP())
  resolver.release()
  assert resolver.latch_active is False
  assert resolver.engaged is False
  params._store[LATERAL_STACK_PARAM] = b"sunnypilot-current"
  current = resolver.current_profile(CP=_FakeCP())
  assert current is not None
  assert current.resolved_lateral == "sunnypilot-current"


def test_active_profile_summary_includes_latch_state():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  resolver.engage(CP=_FakeCP())
  summary = active_profile_summary(resolver)
  assert summary is not None
  assert summary["latched"] is True
  assert summary["engaged"] is True
  assert summary["lateral"]["resolved"] == "custom-2.0"


def test_make_default_resolver_returns_instance():
  r = make_default_resolver()
  assert isinstance(r, ControlsProfileResolver)


def test_current_profile_resolves_immediately_if_no_latch():
  params = _FakeParams({LATERAL_STACK_PARAM: b"custom-2.0", LONGITUDINAL_STACK_PARAM: b"custom-2.0"})
  resolver = ControlsProfileResolver(params_getter=params)
  current = resolver.current_profile(CP=_FakeCP())
  assert current is not None
  assert current.resolved_lateral == "custom-2.0"
