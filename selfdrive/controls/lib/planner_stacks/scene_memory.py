from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openpilot.selfdrive.controls.lib.longitudinal_modes import (
  LongitudinalActuationType,
  LongitudinalMode,
  LongitudinalModeResolution,
)
from openpilot.selfdrive.controls.lib.planner_stacks.selector import SCENE_MEMORY_V1, StackResolution


@dataclass(frozen=True)
class SceneMemorySnapshot:
  enabled: bool = False
  active: bool = False
  shadow: bool = False
  oldest_evidence_age: float = 0.0
  lead_stability: float = 0.0
  path_stability: float = 0.0
  map_speed_stability: float = 0.0
  invalid_evidence_count: int = 0
  stale_evidence_count: int = 0
  provenance: tuple[str, ...] = field(default_factory=tuple)
  source_eligibility: tuple[str, ...] = field(default_factory=tuple)
  summary: str = ""


class SceneMemory:
  """Volatile shadow snapshot over existing planner artifacts.

  Milestone 1 intentionally consumes existing lead/SCC/decision/speed/map state
  instead of classifying sources itself. The snapshot is telemetry-only and must
  not mutate planner output.
  """

  def __init__(self) -> None:
    self.snapshot = SceneMemorySnapshot()

  def reset(self) -> None:
    self.snapshot = SceneMemorySnapshot()

  def update_from_planner(
      self,
      planner: Any,
      mode_resolution: LongitudinalModeResolution | None,
      planner_stack_resolution: StackResolution,
      actuated_stack: str) -> SceneMemorySnapshot:
    requested_scene_memory = planner_stack_resolution.requested_stack == SCENE_MEMORY_V1
    resolved_scene_memory = planner_stack_resolution.resolved_stack == SCENE_MEMORY_V1
    active = actuated_stack == SCENE_MEMORY_V1
    enabled = requested_scene_memory or resolved_scene_memory or active

    provenance = _source_provenance(planner, mode_resolution)
    snapshot = SceneMemorySnapshot(
      enabled=enabled,
      active=active,
      shadow=enabled and not active,
      oldest_evidence_age=0.0,
      lead_stability=_lead_stability(planner),
      path_stability=0.0,
      map_speed_stability=_map_speed_stability(planner),
      invalid_evidence_count=0,
      stale_evidence_count=0,
      provenance=provenance,
      source_eligibility=_source_eligibility(mode_resolution),
      summary=_summary(enabled, active, planner_stack_resolution.fallback_reason),
    )
    self.snapshot = snapshot
    return snapshot


def _bounded_unit(value: object, default: float = 0.0) -> float:
  try:
    value = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return default
  return max(0.0, min(1.0, value))


def _lead_stability(planner: Any) -> float:
  context = getattr(planner, "primary_lead_context", None)
  primary = getattr(context, "behavior", None) or getattr(context, "physical", None)
  if primary is None:
    return 0.0
  return _bounded_unit(getattr(primary, "confidence", 0.0))


def _map_speed_stability(planner: Any) -> float:
  active_resolver = getattr(planner, "active_resolver", None) or getattr(planner, "resolver", None)
  active_sla = getattr(planner, "active_sla", None) or getattr(planner, "sla", None)
  resolver_valid = bool(getattr(active_resolver, "speed_limit_valid", False) or getattr(active_resolver, "speed_limit_last_valid", False))
  assist_active = bool(getattr(active_sla, "is_active", False))
  return 1.0 if resolver_valid or assist_active else 0.0


def _source_provenance(planner: Any, mode_resolution: LongitudinalModeResolution | None) -> tuple[str, ...]:
  provenance: list[str] = ["output:planner"]
  if getattr(planner, "primary_lead_context", None) is not None:
    provenance.append("lead:PrimaryLeadContext")
  if mode_resolution is not None:
    provenance.append("mode:LongitudinalModeResolution")
    provenance.append("scc:SccEvidenceResult")
  if getattr(planner, "longitudinal_decision_telemetry", None) is not None:
    provenance.append("decision:LongitudinalDecisionTelemetry")
  active_resolver = getattr(planner, "active_resolver", None) or getattr(planner, "resolver", None)
  if active_resolver is not None and (
      getattr(active_resolver, "speed_limit_valid", False) or getattr(active_resolver, "speed_limit_last_valid", False)):
    provenance.append("speed:SpeedLimitResolver")
  active_sla = getattr(planner, "active_sla", None) or getattr(planner, "sla", None)
  if active_sla is not None and getattr(active_sla, "is_active", False):
    provenance.append("speed:SpeedLimitAssist")
  active_scc = getattr(planner, "active_scc", None) or getattr(planner, "scc", None)
  scc_vision = getattr(active_scc, "vision", None)
  scc_map = getattr(active_scc, "map", None)
  if active_scc is not None and (
      getattr(scc_vision, "is_active", False) or getattr(scc_vision, "is_enabled", False) or
      getattr(scc_map, "is_active", False) or getattr(scc_map, "is_enabled", False)):
    provenance.append("curve:SmartCruiseControl")
  osm_traffic_control_prior = getattr(planner, "osm_traffic_control_prior", None)
  if osm_traffic_control_prior is not None and getattr(osm_traffic_control_prior, "active", False):
    provenance.append("map:OsmTrafficControlPrior")
  return tuple(provenance)


def _source_eligibility(mode_resolution: LongitudinalModeResolution | None) -> tuple[str, ...]:
  if mode_resolution is None:
    return ("cruise", "lead")

  eligible = ["cruise", "lead"]
  if mode_resolution.actuation_type == LongitudinalActuationType.SET_SPEED_ADVISORY:
    eligible.append("set_speed_advisory")
    return tuple(eligible)

  if mode_resolution.requested_mode == LongitudinalMode.E2E or mode_resolution.e2e_like:
    eligible.append("model_stop")
  if mode_resolution.requested_mode == LongitudinalMode.SCC:
    eligible.extend(("scc_evidence", "scc_curve", "speed_limit", "map_caution", "osm"))
  return tuple(dict.fromkeys(eligible))


def _summary(enabled: bool, active: bool, fallback_reason: str) -> str:
  if active:
    return "scene_memory_active"
  if enabled:
    return fallback_reason or "scene_memory_shadow"
  return "planner_current"
