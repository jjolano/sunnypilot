from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from openpilot.selfdrive.controls.lib.scc_evidence import (
  SccAdvisoryFlags,
  SccEvidenceAdvisory,
  SccEvidenceResult,
  SccEvidenceTier,
  SccModeEvidence,
)


LONGITUDINAL_MODE_MIGRATION_VERSION = "1.1"

LONGITUDINAL_MODE_PARAM = "LongitudinalMode"
LONGITUDINAL_MODE_MIGRATION_PARAM = "LongitudinalModeMigrationVersion"
SCC_CURVE_VISION_PARAM = "SccCurveVisionEnabled"
SCC_CURVE_MAP_PARAM = "SccCurveMapEnabled"

LEGACY_LONGITUDINAL_MODE_PARAMS = frozenset((
  "DynamicExperimentalControl",
  "SmartCruiseControlVision",
  "SmartCruiseControlMap",
  "ExperimentalMode",
))

LEGACY_SCC_CURVE_PARAM_MIGRATIONS = {
  "SmartCruiseControlVision": SCC_CURVE_VISION_PARAM,
  "SmartCruiseControlMap": SCC_CURVE_MAP_PARAM,
}


class LongitudinalMode(IntEnum):
  ACC = 0
  E2E = 1
  SCC = 2


class ResolvedLongitudinalImplementation(Enum):
  HARDWARE_ACC = "hardware_acc"
  MODEL_ACC = "model_acc"
  E2E = "e2e"
  SCC_ACC = "scc_acc"
  SCC_E2E = "scc_e2e"
  ICBM_ADVISORY = "icbm_advisory"


class LongitudinalActuationType(Enum):
  DIRECT = "direct"
  SET_SPEED_ADVISORY = "set_speed_advisory"


class DecCompatibilityState(Enum):
  ACC = "acc"
  BLENDED = "blended"


@dataclass(frozen=True)
class LongitudinalModeResolution:
  requested_mode: LongitudinalMode
  resolved_implementation: ResolvedLongitudinalImplementation
  actuation_type: LongitudinalActuationType
  restriction_status: tuple[str, ...] = ()
  compatibility_alias_state: DecCompatibilityState = DecCompatibilityState.ACC
  unsupported_reason: str = ""
  debug: dict[str, str] = field(default_factory=dict)
  scc_evidence: SccEvidenceResult = field(default_factory=SccEvidenceResult)

  @property
  def acc_like(self) -> bool:
    return self.resolved_implementation in (
      ResolvedLongitudinalImplementation.HARDWARE_ACC,
      ResolvedLongitudinalImplementation.MODEL_ACC,
      ResolvedLongitudinalImplementation.SCC_ACC,
      ResolvedLongitudinalImplementation.ICBM_ADVISORY,
    )

  @property
  def e2e_like(self) -> bool:
    return self.resolved_implementation in (
      ResolvedLongitudinalImplementation.E2E,
      ResolvedLongitudinalImplementation.SCC_E2E,
    )


def _decode_param_value(value: Any) -> Any:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return value


def _param_get(params: Any, key: str, default: Any = None) -> Any:
  get = getattr(params, "get", None)
  if get is None:
    return default
  try:
    value = get(key)
  except TypeError:
    value = get(key, default)
  return default if value is None else _decode_param_value(value)


def _param_get_bool(params: Any, key: str) -> bool:
  get_bool = getattr(params, "get_bool", None)
  if get_bool is not None:
    try:
      return bool(get_bool(key))
    except Exception:
      return False
  value = _param_get(params, key)
  if isinstance(value, bool):
    return value
  if value is None:
    return False
  return str(value).lower() in ("1", "true", "yes")


def _param_put(params: Any, key: str, value: Any) -> None:
  put = getattr(params, "put", None)
  if put is None:
    raise AttributeError("params object does not support put")
  put(key, value)


def _parse_mode(value: Any) -> LongitudinalMode | None:
  value = _decode_param_value(value)
  if isinstance(value, LongitudinalMode):
    return value
  try:
    return LongitudinalMode(int(value))
  except (TypeError, ValueError):
    return None


def longitudinal_mode_migration_current(params: Any) -> bool:
  return _param_get(params, LONGITUDINAL_MODE_MIGRATION_PARAM) == LONGITUDINAL_MODE_MIGRATION_VERSION


def longitudinal_mode_source_of_truth_exists(params: Any) -> bool:
  return _param_get(params, LONGITUDINAL_MODE_PARAM) is not None


def legacy_longitudinal_mode_params_ignored(params: Any) -> bool:
  return longitudinal_mode_migration_current(params)


def longitudinal_mode_degraded_reason(params: Any) -> str:
  if longitudinal_mode_migration_current(params) and not longitudinal_mode_source_of_truth_exists(params):
    return "migration_current_missing_mode"
  return ""


def requested_mode_from_params(params: Any) -> LongitudinalMode:
  parsed = _parse_mode(_param_get(params, LONGITUDINAL_MODE_PARAM))
  if parsed is not None:
    return parsed

  if longitudinal_mode_migration_current(params):
    return LongitudinalMode.ACC

  # Centralized compatibility fallback for pre-migration runtimes/tests only.
  experimental = _param_get_bool(params, "ExperimentalMode")
  dec = _param_get_bool(params, "DynamicExperimentalControl")
  if experimental and dec:
    return LongitudinalMode.SCC
  if experimental:
    return LongitudinalMode.E2E
  return LongitudinalMode.ACC


def migrate_longitudinal_mode_params(params: Any) -> bool:
  if longitudinal_mode_migration_current(params):
    return False

  migrated = False
  for legacy_key, target_key in LEGACY_SCC_CURVE_PARAM_MIGRATIONS.items():
    if _param_get(params, legacy_key) is not None and _param_get(params, target_key) is None:
      _param_put(params, target_key, _param_get_bool(params, legacy_key))
      migrated = True

  if not longitudinal_mode_source_of_truth_exists(params):
    _param_put(params, LONGITUDINAL_MODE_PARAM, int(requested_mode_from_params(params)))
    migrated = True

  _param_put(params, LONGITUDINAL_MODE_MIGRATION_PARAM, LONGITUDINAL_MODE_MIGRATION_VERSION)
  return migrated


def filter_legacy_longitudinal_mode_params(params_to_update: dict[str, Any], params: Any) -> dict[str, Any]:
  if not legacy_longitudinal_mode_params_ignored(params):
    return dict(params_to_update)
  return {key: value for key, value in params_to_update.items() if key not in LEGACY_LONGITUDINAL_MODE_PARAMS}


def should_skip_legacy_longitudinal_restore(param: str, params: Any) -> bool:
  return param in LEGACY_LONGITUDINAL_MODE_PARAMS and legacy_longitudinal_mode_params_ignored(params)


def _has_direct_longitudinal_control(CP: Any | None) -> bool:
  return bool(getattr(CP, "openpilotLongitudinalControl", True))


def _radar_unavailable(CP: Any | None) -> bool:
  return bool(getattr(CP, "radarUnavailable", False))


def resolve_longitudinal_mode(params: Any, CP: Any | None = None, *,
                              scc_evidence: SccModeEvidence | SccEvidenceResult | None = None,
                              unsupported_reason: str = "", restriction_status: tuple[str, ...] = ()) -> LongitudinalModeResolution:
  requested = requested_mode_from_params(params)
  actuation = LongitudinalActuationType.DIRECT if _has_direct_longitudinal_control(CP) else LongitudinalActuationType.SET_SPEED_ADVISORY
  degraded_reason = longitudinal_mode_degraded_reason(params)
  effective_unsupported_reason = unsupported_reason or degraded_reason
  effective_restriction_status = restriction_status
  if degraded_reason and degraded_reason not in effective_restriction_status:
    effective_restriction_status = (*effective_restriction_status, degraded_reason)

  if actuation == LongitudinalActuationType.SET_SPEED_ADVISORY:
    return LongitudinalModeResolution(
      requested_mode=requested,
      resolved_implementation=ResolvedLongitudinalImplementation.ICBM_ADVISORY,
      actuation_type=actuation,
      restriction_status=effective_restriction_status,
      compatibility_alias_state=DecCompatibilityState.ACC,
      unsupported_reason=effective_unsupported_reason,
      debug={"reason": effective_unsupported_reason or "set_speed_advisory"},
    )

  if requested == LongitudinalMode.E2E:
    return LongitudinalModeResolution(
      requested_mode=requested,
      resolved_implementation=ResolvedLongitudinalImplementation.E2E,
      actuation_type=actuation,
      restriction_status=effective_restriction_status,
      compatibility_alias_state=DecCompatibilityState.BLENDED,
      unsupported_reason=effective_unsupported_reason,
      debug={"reason": effective_unsupported_reason or "requested_e2e"},
    )

  if requested == LongitudinalMode.SCC:
    scc_evidence = scc_evidence or SccModeEvidence()
    scc_evidence_result = _classify_scc_mode_evidence(scc_evidence)
    scc_e2e_active = scc_evidence_result.e2e_active
    resolved = ResolvedLongitudinalImplementation.SCC_E2E if scc_e2e_active else ResolvedLongitudinalImplementation.SCC_ACC
    return LongitudinalModeResolution(
      requested_mode=requested,
      resolved_implementation=resolved,
      actuation_type=actuation,
      restriction_status=effective_restriction_status,
      compatibility_alias_state=DecCompatibilityState.BLENDED if scc_e2e_active else DecCompatibilityState.ACC,
      unsupported_reason=effective_unsupported_reason,
      debug={"reason": effective_unsupported_reason or _scc_evidence_debug_reason(scc_evidence, scc_evidence_result)},
      scc_evidence=scc_evidence_result,
    )

  resolved = ResolvedLongitudinalImplementation.MODEL_ACC if _radar_unavailable(CP) else ResolvedLongitudinalImplementation.HARDWARE_ACC
  return LongitudinalModeResolution(
    requested_mode=LongitudinalMode.ACC,
    resolved_implementation=resolved,
    actuation_type=actuation,
    restriction_status=effective_restriction_status,
    compatibility_alias_state=DecCompatibilityState.ACC,
    unsupported_reason=effective_unsupported_reason,
    debug={"reason": effective_unsupported_reason or resolved.value},
  )


class LongitudinalModeResolver:
  @staticmethod
  def resolve(params: Any, CP: Any | None = None, *,
              scc_evidence: SccModeEvidence | SccEvidenceResult | None = None,
              unsupported_reason: str = "", restriction_status: tuple[str, ...] = ()) -> LongitudinalModeResolution:
    return resolve_longitudinal_mode(
      params, CP,
      scc_evidence=scc_evidence,
      unsupported_reason=unsupported_reason,
      restriction_status=restriction_status,
    )


def _classify_scc_mode_evidence(scc_evidence: SccModeEvidence | SccEvidenceResult) -> SccEvidenceResult:
  if isinstance(scc_evidence, SccEvidenceResult):
    return scc_evidence
  return scc_evidence.classify()


def _scc_evidence_debug_reason(scc_evidence: SccModeEvidence | SccEvidenceResult,
                               result: SccEvidenceResult) -> str:
  if result.e2e_active:
    return result.reason
  if result.reason in ("scc_e2e_pending", "scc_acc_recovery"):
    return result.reason
  if result.confirmed_lead:
    return "scc_confirmed_lead"
  return getattr(scc_evidence, "reason", result.reason)
