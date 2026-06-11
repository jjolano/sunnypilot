from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpilot.selfdrive.controls.lib.policy_facade import (
  ControlsProfile,
  policy_summary,
  resolve_controls_profile,
)


LATERAL_STACK_PARAM = "LateralStack"
LONGITUDINAL_STACK_PARAM = "LongitudinalStack"


@dataclass
class EngagementState:
  engaged: bool = False
  latch_active: bool = False
  latch_profile: ControlsProfile | None = None


class ControlsProfileResolver:
  """Per-engagement resolver that latches the active controls profile.

  On `engage()`, the resolver reads the current user-selected stack
  values, runs them through the policy facade, and locks the resulting
  profile for the duration of the engagement. On `release()`, the
  latch is cleared and the next call to `current_profile()` re-queries
  the user params.

  Latching prevents the active stack identity from swapping mid-engage
  (for example if a UI toggle changes while driving), which would
  otherwise look like a stack identity change in telemetry and on
  consumers downstream of controlsState.
  """

  def __init__(self, params_getter=None) -> None:
    self._params_getter = params_getter or _default_params_getter
    self._state = EngagementState()
    self._last_unlatched_profile: ControlsProfile | None = None

  def resolve(self, *, CP: object | None = None, CP_SP: object | None = None,
              manifest: dict[str, Any] | None = None) -> ControlsProfile:
    requested_lateral, requested_longitudinal = self._read_user_stacks()
    profile = resolve_controls_profile(
      requested_lateral, requested_longitudinal, CP=CP, CP_SP=CP_SP, manifest=manifest,
    )
    self._last_unlatched_profile = profile
    return profile

  def engage(self, *, CP: object | None = None, CP_SP: object | None = None,
             manifest: dict[str, Any] | None = None) -> ControlsProfile:
    profile = self.resolve(CP=CP, CP_SP=CP_SP, manifest=manifest)
    self._state.engaged = True
    self._state.latch_active = True
    self._state.latch_profile = profile
    return profile

  def release(self) -> None:
    self._state.engaged = False
    self._state.latch_active = False
    self._state.latch_profile = None

  def current_profile(self, *, CP: object | None = None, CP_SP: object | None = None,
                      manifest: dict[str, Any] | None = None) -> ControlsProfile | None:
    if self._state.latch_active and self._state.latch_profile is not None:
      return self._state.latch_profile
    return self.resolve(CP=CP, CP_SP=CP_SP, manifest=manifest)

  @property
  def engaged(self) -> bool:
    return self._state.engaged

  @property
  def latch_active(self) -> bool:
    return self._state.latch_active

  def _read_user_stacks(self) -> tuple[str, str]:
    lateral = self._params_getter(LATERAL_STACK_PARAM, "") or ""
    longitudinal = self._params_getter(LONGITUDINAL_STACK_PARAM, "") or ""
    return (lateral, longitudinal)


def _default_params_getter(key: str, default: str) -> str:
  try:
    from openpilot.common.params import Params
    return Params().get(key, encoding="utf-8") or default
  except Exception:
    return default


def make_default_resolver() -> ControlsProfileResolver:
  return ControlsProfileResolver()


def active_profile_summary(resolver: ControlsProfileResolver) -> dict[str, Any] | None:
  profile = resolver.current_profile()
  if profile is None:
    return None
  summary = policy_summary(profile)
  summary["latched"] = resolver.latch_active
  summary["engaged"] = resolver.engaged
  return summary
