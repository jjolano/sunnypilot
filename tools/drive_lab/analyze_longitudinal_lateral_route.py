#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, cast

import numpy as np

from openpilot.tools.drive_lab.route_analysis import (
  LATERAL_DEMAND_SCHEMA_LEGACY,
  conditioned_desired_curvature,
  lateral_demand_schema,
)
from openpilot.tools.drive_lab.timeline import format_enum, msg_payload, msg_time_s, msg_type
from openpilot.tools.lib.logreader import LogReader, ReadMode


DEFAULT_LOG_ROOTS = (
  Path("/tmp/opencode/longitudinal-log-analysis"),
  Path("/tmp/opencode/lateral-log-analysis"),
  Path("/tmp/opencode/device-latest-two-routes"),
  Path("/tmp/opencode/route-analysis"),
)
QLOG_NAMES = ("qlog.zst", "qlog.bz2")
RLOG_NAMES = ("rlog.zst", "rlog.bz2")


COLUMNS = (
  "time_s",
  "msg_type",
  "selfdrive_active",
  "carcontrol_long_active",
  "carcontrol_lat_active",
  "v_ego",
  "a_ego",
  "gas_pressed",
  "brake_pressed",
  "standstill",
  "steering_angle_deg",
  "steering_pressed",
  "steering_torque",
  "carcontrol_accel",
  "caroutput_accel",
  "carcontrol_torque",
  "caroutput_torque",
  "carcontrol_curvature",
  "controls_long_state",
  "controls_force_decel",
  "controls_curvature",
  "controls_desired_curvature",
  "lat_state_kind",
  "lat_state_active",
  "lat_state_output",
  "lat_state_saturated",
  "lat_desired_lateral_accel",
  "lat_actual_lateral_accel",
  "lat_error",
  "lat_governor_reason",
  "lat_sample_reject_reason",
  "lat_learned_factor",
  "lat_learned_friction",
  "lat_learned_response_delay",
  "lat_actual_lateral_jerk",
  "plan_a_target",
  "plan_should_stop",
  "plan_has_lead",
  "plan_source",
  "plan_fcw",
  "sp_a_target",
  "sp_source",
  "sp_stack_requested",
  "sp_stack_resolved",
  "sp_stack_actuated",
  "sp_selected_intent",
  "sp_selected_reason",
  "sp_seed_context",
  "sp_seed_candidate",
  "sp_decision_enabled",
  "sp_decision_raw_source",
  "sp_decision_raw_reason",
  "sp_decision_applied_reason",
  "sp_decision_raw_a_target",
  "sp_decision_applied_a_target",
  "sp_long_mode_requested",
  "sp_long_mode_impl",
  "sp_long_mode_evidence_tier",
  "sp_long_mode_evidence_reason",
  "lead0_status",
  "lead0_d_rel",
  "lead0_v_rel",
  "lead0_v_lead",
  "lead0_v_lead_k",
  "lead0_a_lead_k",
  "lead0_y_rel",
  "lead0_model_prob",
  "lead0_radar_track_id",
  "lead1_status",
  "lead1_d_rel",
  "lead1_v_rel",
  "lead1_v_lead",
  "lead1_v_lead_k",
  "lead1_a_lead_k",
  "lead1_y_rel",
  "lead1_model_prob",
  "lead1_radar_track_id",
  "model_action_should_stop",
  "model_action_desired_accel",
  "model_action_desired_curvature",
  "model_position_x_last",
  "model_velocity_x_last",
  "model_position_y_last",
  "model_orientation_z_last",
  "lateral_plan_lane_change_state",
  "lateral_plan_lane_change_direction",
  "lateral_plan_curvature0",
  "lateral_plan_raw_curvature",
  "lateral_plan_processed_curvature",
  "model_path_active",
  "model_path_gated",
  "model_path_quality",
  "model_path_reason",
  "model_path_raw_curvature",
  "model_path_processed_curvature",
  "model_path_lane_change_fade",
  "model_path_lane_rate_damping_mode",
  "model_path_lane_rate_damping_active",
  "model_path_lane_rate_damping_applied",
  "model_path_lane_rate_damping_reason",
  "model_path_lane_rate_damping_lane_center",
  "model_path_lane_rate_damping_lane_center_rate",
  "model_path_lane_rate_damping_lat_accel",
  "model_path_lane_rate_damping_curvature",
  "model_path_lane_rate_damping_cap_lat_accel",
  "model_path_lane_fit_source_mode",
  "model_path_lane_fit_source_active",
  "model_path_lane_fit_source_applied",
  "model_path_lane_fit_source_reason",
  "model_path_lane_fit_source_candidate_curvature",
  "model_path_lane_fit_source_applied_curvature",
  "model_path_lane_fit_source_lat_accel_delta",
  "model_path_lane_fit_source_confidence",
  "model_path_lane_fit_source_slew_limited",
)


@dataclass(frozen=True)
class AnalysisWindow:
  start_s: float
  end_s: float
  label: str


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Extract aligned longitudinal/lateral route-window signals from qlogs/rlogs."
  )
  parser.add_argument("inputs", nargs="+", help="Route id/name, local route dir, log file, URL, or LogReader route string")
  parser.add_argument("--segment", type=int, help="Local short-route segment to load; LogReader routes use route/segment syntax")
  parser.add_argument("--start", type=float, required=True, help="Window start in seconds from first loaded message")
  parser.add_argument("--end", type=float, required=True, help="Window end in seconds from first loaded message")
  parser.add_argument("--label", default="window", help="Label used in the text report")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs when resolving route inputs")
  parser.add_argument("--rlog", action="store_true", help="Prefer rlogs when resolving route inputs")
  parser.add_argument("--csv", dest="csv_path", help="Write aligned per-frame/window CSV to this path")
  parser.add_argument("--log-root", action="append", default=[], help="Extra root to search for local short-route logs")
  args = parser.parse_args()

  if args.start > args.end:
    raise ValueError("--start must be <= --end")
  if args.qlog and args.rlog:
    raise ValueError("choose at most one of --qlog or --rlog")

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.RLOG if args.rlog else ReadMode.AUTO
  log_roots = tuple(Path(p) for p in args.log_root) + DEFAULT_LOG_ROOTS
  identifiers = resolve_inputs(args.inputs, segment=args.segment, read_mode=read_mode, log_roots=log_roots)
  msgs = list(LogReader(identifiers, default_mode=read_mode, sort_by_time=True))
  window = AnalysisWindow(args.start, args.end, args.label)
  rows = extract_rows(msgs, window)
  if args.csv_path:
    write_csv(rows, args.csv_path)
  print(render_report(rows, window, identifiers))


def resolve_inputs(inputs: list[str], *, segment: int | None, read_mode: ReadMode, log_roots: tuple[Path, ...]) -> list[str]:
  resolved: list[str] = []
  for value in inputs:
    path = Path(value).expanduser()
    if path.is_file():
      resolved.append(str(path))
      continue
    if path.is_dir():
      resolved.extend(_logs_in_dir(path, segment=segment, read_mode=read_mode))
      continue

    local = _resolve_local_short_route(value, segment=segment, read_mode=read_mode, log_roots=log_roots)
    if local:
      resolved.extend(local)
      continue

    if segment is not None and "--" not in value.rsplit("/", 1)[-1]:
      resolved.append(f"{value}/{segment}")
    else:
      resolved.append(value)

  if not resolved:
    raise ValueError("no logs resolved")
  return _dedupe_sorted(resolved)


def _logs_in_dir(path: Path, *, segment: int | None, read_mode: ReadMode) -> list[str]:
  names = _preferred_log_names(read_mode)
  files: list[Path] = []
  if segment is not None:
    segment_suffix = f"--{segment}"
    for child in sorted(path.rglob("*")):
      if child.is_dir() and child.name.endswith(segment_suffix):
        files.extend(_log_files_in_segment_dir(child, names))
  else:
    for name in names:
      files.extend(path.rglob(name))
  return [str(p) for p in sorted(files, key=_natural_path_key)]


def _resolve_local_short_route(route: str, *, segment: int | None, read_mode: ReadMode, log_roots: tuple[Path, ...]) -> list[str]:
  names = _preferred_log_names(read_mode)
  matches: list[Path] = []
  if segment is not None:
    segment_dirs = (f"{route}--{segment}",)
  else:
    segment_dirs = ()
  for root in log_roots:
    if not root.exists():
      continue
    if segment is None:
      for name in names:
        matches.extend(root.rglob(f"{route}--*/{name}"))
    else:
      for segment_dir in segment_dirs:
        for name in names:
          matches.extend(root.rglob(f"{segment_dir}/{name}"))
  return [str(p) for p in sorted(_dedupe_segment_logs(matches), key=_natural_path_key)]


def _dedupe_segment_logs(paths: list[Path]) -> list[Path]:
  selected: dict[tuple[str, str], Path] = {}
  for path in paths:
    key = (path.parent.name, path.name)
    selected.setdefault(key, path)
  return list(selected.values())


def _log_files_in_segment_dir(path: Path, names: tuple[str, ...]) -> list[Path]:
  return [path / name for name in names if (path / name).exists()]


def _preferred_log_names(read_mode: ReadMode) -> tuple[str, ...]:
  if read_mode == ReadMode.QLOG:
    return QLOG_NAMES
  if read_mode == ReadMode.RLOG:
    return RLOG_NAMES
  return RLOG_NAMES + QLOG_NAMES


def _dedupe_sorted(values: list[str]) -> list[str]:
  return sorted(dict.fromkeys(values), key=_natural_path_key)


def _natural_path_key(value: str | Path) -> tuple[Any, ...]:
  text = str(value)
  parts: list[Any] = []
  cur = ""
  for ch in text:
    if ch.isdigit():
      cur += ch
    else:
      if cur:
        parts.append(int(cur))
        cur = ""
      parts.append(ch)
  if cur:
    parts.append(int(cur))
  return tuple(parts)


def extract_rows(msgs: list[Any], window: AnalysisWindow) -> list[dict[str, Any]]:
  if not msgs:
    return []
  ordered = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  demand_schema = lateral_demand_schema(ordered)
  base_mono_time = int(getattr(ordered[0], "logMonoTime", 0))
  state: dict[str, Any] = {column: "" for column in COLUMNS}
  rows: list[dict[str, Any]] = []

  for msg in ordered:
    t = msg_time_s(msg, base_mono_time)
    typ = msg_type(msg)
    if t > window.end_s:
      break
    payload = msg_payload(msg)
    if _is_relevant_message(typ):
      update_state(state, typ, payload, demand_schema)
      if window.start_s <= t <= window.end_s:
        row = dict(state)
        row["time_s"] = round(t, 3)
        row["msg_type"] = typ
        rows.append(row)
  return rows


def _is_relevant_message(typ: str) -> bool:
  return typ in {
    "selfdriveState",
    "carState",
    "carControl",
    "carOutput",
    "controlsState",
    "longitudinalPlan",
    "longitudinalPlanSP",
    "radarState",
    "modelV2",
    "lateralPlan",
    "liveTorqueParameters",
  }


def update_state(state: dict[str, Any], typ: str, payload: Any, demand_schema: str = LATERAL_DEMAND_SCHEMA_LEGACY) -> None:
  if typ == "selfdriveState":
    _set(state, "selfdrive_active", _safe_get(payload, "active"))
  elif typ == "carState":
    for column, path in (
      ("v_ego", "vEgo"),
      ("a_ego", "aEgo"),
      ("gas_pressed", "gasPressed"),
      ("brake_pressed", "brakePressed"),
      ("standstill", "standstill"),
      ("steering_angle_deg", "steeringAngleDeg"),
      ("steering_pressed", "steeringPressed"),
      ("steering_torque", "steeringTorque"),
    ):
      _set(state, column, _safe_get(payload, path))
  elif typ == "carControl":
    for column, path in (
      ("carcontrol_long_active", "longActive"),
      ("carcontrol_lat_active", "latActive"),
      ("carcontrol_accel", "actuators.accel"),
      ("carcontrol_torque", "actuators.torque"),
      ("carcontrol_curvature", "actuators.curvature"),
    ):
      _set(state, column, _safe_get(payload, path))
  elif typ == "carOutput":
    for column, path in (
      ("caroutput_accel", "actuatorsOutput.accel"),
      ("caroutput_torque", "actuatorsOutput.torque"),
    ):
      _set(state, column, _safe_get(payload, path))
  elif typ == "controlsState":
    _set(state, "controls_long_state", _enum(_safe_get(payload, "longControlState")))
    _set(state, "controls_force_decel", _safe_get(payload, "forceDecel"))
    _set(state, "controls_curvature", _safe_get(payload, "curvature"))
    _set(state, "controls_desired_curvature", _safe_get(payload, "desiredCurvature"))
    _update_lateral_control_state(state, payload)
    _update_model_path_state(state, _safe_get(payload, "modelPathState"), demand_schema)
  elif typ == "longitudinalPlan":
    for column, path in (
      ("plan_a_target", "aTarget"),
      ("plan_should_stop", "shouldStop"),
      ("plan_has_lead", "hasLead"),
      ("plan_fcw", "fcw"),
    ):
      _set(state, column, _safe_get(payload, path))
    _set(state, "plan_source", _enum(_safe_get(payload, "longitudinalPlanSource")))
  elif typ == "longitudinalPlanSP":
    _set(state, "sp_a_target", _safe_get(payload, "aTarget"))
    _set(state, "sp_source", _enum(_safe_get(payload, "longitudinalPlanSource")))
    for column, path in (
      ("sp_stack_requested", "stack.requestedStack"),
      ("sp_stack_resolved", "stack.resolvedStack"),
      ("sp_stack_actuated", "stack.actuatedStack"),
      ("sp_selected_intent", "stack.selectedIntent"),
      ("sp_selected_reason", "stack.selectedReason"),
      ("sp_seed_context", "stack.seedContext"),
      ("sp_seed_candidate", "stack.seedCandidate"),
      ("sp_decision_enabled", "decisionLayer.enabled"),
      ("sp_decision_raw_source", "decisionLayer.rawSource"),
      ("sp_decision_raw_reason", "decisionLayer.rawReason"),
      ("sp_decision_applied_reason", "decisionLayer.appliedReason"),
      ("sp_decision_raw_a_target", "decisionLayer.rawATarget"),
      ("sp_decision_applied_a_target", "decisionLayer.appliedATarget"),
      ("sp_long_mode_requested", "longitudinalMode.requestedMode"),
      ("sp_long_mode_impl", "longitudinalMode.resolvedImplementation"),
      ("sp_long_mode_evidence_tier", "longitudinalMode.evidenceTier"),
      ("sp_long_mode_evidence_reason", "longitudinalMode.evidenceReason"),
    ):
      value = _safe_get(payload, path)
      _set(state, column, _enum(value) if column in {"sp_stack_requested", "sp_stack_resolved", "sp_stack_actuated", "sp_long_mode_requested", "sp_long_mode_impl"} else value)
  elif typ == "radarState":
    _update_lead(state, "lead0", _safe_get(payload, "leadOne"))
    _update_lead(state, "lead1", _safe_get(payload, "leadTwo"))
  elif typ == "modelV2":
    for column, path in (
      ("model_action_should_stop", "action.shouldStop"),
      ("model_action_desired_accel", "action.desiredAcceleration"),
      ("model_action_desired_curvature", "action.desiredCurvature"),
    ):
      _set(state, column, _safe_get(payload, path))
    _set(state, "model_position_x_last", _last_number(_safe_get(payload, "position.x")))
    _set(state, "model_velocity_x_last", _last_number(_safe_get(payload, "velocity.x")))
    _set(state, "model_position_y_last", _last_number(_safe_get(payload, "position.y")))
    _set(state, "model_orientation_z_last", _last_number(_safe_get(payload, "orientation.z")))
  elif typ == "lateralPlan":
    _set(state, "lateral_plan_lane_change_state", _enum(_safe_get(payload, "laneChangeState")))
    _set(state, "lateral_plan_lane_change_direction", _enum(_safe_get(payload, "laneChangeDirection")))
    _set(state, "lateral_plan_curvature0", _first_number(_safe_get(payload, "curvatures")))
    _set(state, "lateral_plan_raw_curvature", _safe_get(payload, "rawCurvature"))
    _set(state, "lateral_plan_processed_curvature", _safe_get(payload, "curvature"))
  elif typ == "liveTorqueParameters":
    _set(state, "lat_learned_factor", _safe_get(payload, "latAccelFactor"))
    _set(state, "lat_learned_friction", _safe_get(payload, "friction"))


def _update_lead(state: dict[str, Any], prefix: str, lead: Any) -> None:
  status = bool(_safe_get(lead, "status", False))
  _set(state, f"{prefix}_status", status)
  if not status:
    for suffix in ("d_rel", "v_rel", "v_lead", "v_lead_k", "a_lead_k", "y_rel", "model_prob", "radar_track_id"):
      state[f"{prefix}_{suffix}"] = ""
    return
  for suffix, path in (
    ("d_rel", "dRel"),
    ("v_rel", "vRel"),
    ("v_lead", "vLead"),
    ("v_lead_k", "vLeadK"),
    ("a_lead_k", "aLeadK"),
    ("y_rel", "yRel"),
    ("model_prob", "modelProb"),
    ("radar_track_id", "radarTrackId"),
  ):
    _set(state, f"{prefix}_{suffix}", _safe_get(lead, path))


def _update_model_path_state(state: dict[str, Any], path_state: Any, demand_schema: str) -> None:
  for column, path in (
    ("model_path_active", "active"),
    ("model_path_gated", "gated"),
    ("model_path_quality", "quality"),
    ("model_path_reason", "reason"),
    ("model_path_raw_curvature", "rawDesiredCurvature"),
    ("model_path_lane_change_fade", "laneChangeFade"),
    ("model_path_lane_rate_damping_mode", "laneRateDampingMode"),
    ("model_path_lane_rate_damping_active", "laneRateDampingActive"),
    ("model_path_lane_rate_damping_applied", "laneRateDampingApplied"),
    ("model_path_lane_rate_damping_reason", "laneRateDampingReason"),
    ("model_path_lane_rate_damping_lane_center", "laneRateDampingLaneCenter"),
    ("model_path_lane_rate_damping_lane_center_rate", "laneRateDampingLaneCenterRate"),
    ("model_path_lane_rate_damping_lat_accel", "laneRateDampingLatAccel"),
    ("model_path_lane_rate_damping_curvature", "laneRateDampingCurvature"),
    ("model_path_lane_rate_damping_cap_lat_accel", "laneRateDampingCapLatAccel"),
    ("model_path_lane_fit_source_mode", "laneFitSourceMode"),
    ("model_path_lane_fit_source_active", "laneFitSourceActive"),
    ("model_path_lane_fit_source_applied", "laneFitSourceApplied"),
    ("model_path_lane_fit_source_reason", "laneFitSourceReason"),
    ("model_path_lane_fit_source_candidate_curvature", "laneFitSourceCandidateCurvature"),
    ("model_path_lane_fit_source_applied_curvature", "laneFitSourceAppliedCurvature"),
    ("model_path_lane_fit_source_lat_accel_delta", "laneFitSourceLatAccelDelta"),
    ("model_path_lane_fit_source_confidence", "laneFitSourceConfidence"),
    ("model_path_lane_fit_source_slew_limited", "laneFitSourceSlewLimited"),
  ):
    value = _safe_get(path_state, path)
    _set(state, column, _enum(value) if column == "model_path_reason" else value)
  _set(state, "model_path_processed_curvature", conditioned_desired_curvature(path_state, demand_schema))


def _update_lateral_control_state(state: dict[str, Any], payload: Any) -> None:
  lat_state = _safe_get(payload, "lateralControlState")
  kind = _union_which(lat_state)
  _set(state, "lat_state_kind", kind)
  selected = _safe_get(lat_state, kind) if kind else None
  for column, path in (
    ("lat_state_active", "active"),
    ("lat_state_output", "output"),
    ("lat_state_saturated", "saturated"),
    ("lat_desired_lateral_accel", "desiredLateralAccel"),
    ("lat_actual_lateral_accel", "actualLateralAccel"),
    ("lat_error", "error"),
  ):
    _set(state, column, _safe_get(selected, path))
  adaptive = _safe_get(selected, "adaptiveTorqueState")
  for column, path in (
    ("lat_governor_reason", "governorReason"),
    ("lat_sample_reject_reason", "sampleRejectReason"),
    ("lat_learned_factor", "learnedLatAccelFactor"),
    ("lat_learned_friction", "learnedFriction"),
    ("lat_learned_response_delay", "learnedResponseDelay"),
    ("lat_actual_lateral_jerk", "actualLateralJerk"),
  ):
    _set(state, column, _safe_get(adaptive, path))


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(COLUMNS), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(cast(list[dict[str, Any]], rows))


def render_report(rows: list[dict[str, Any]], window: AnalysisWindow, identifiers: list[str]) -> str:
  lines = [
    f"Route window analysis: {window.label}",
    f"window: {window.start_s:.1f}-{window.end_s:.1f}s",
    f"logs: {len(identifiers)}",
    f"samples: {len(rows)}",
  ]
  if not rows:
    lines.append("no samples in window")
    return "\n".join(lines)

  lines.extend(("", "Longitudinal:"))
  lines.extend(_render_longitudinal(rows))
  lines.extend(("", "Lateral:"))
  lines.extend(_render_lateral(rows))
  return "\n".join(lines)


def _render_longitudinal(rows: list[dict[str, Any]]) -> list[str]:
  lines: list[str] = []
  for column, label in (
    ("v_ego", "vEgo"),
    ("a_ego", "aEgo"),
    ("plan_a_target", "aTarget"),
    ("carcontrol_accel", "carControl.accel"),
    ("caroutput_accel", "carOutput.accel"),
    ("lead0_d_rel", "lead0.dRel"),
    ("lead0_v_rel", "lead0.vRel"),
    ("lead0_a_lead_k", "lead0.aLeadK"),
  ):
    stat = _stats(rows, column)
    if stat:
      lines.append(f"  {label}: {stat}")

  for column, label, threshold in (
    ("plan_a_target", "aTarget <= -0.2", -0.2),
    ("plan_a_target", "aTarget <= -0.5", -0.5),
    ("plan_a_target", "aTarget <= -1.0", -1.0),
    ("a_ego", "aEgo <= -0.2", -0.2),
    ("a_ego", "aEgo <= -0.5", -0.5),
    ("carcontrol_accel", "actuator <= -0.2", -0.2),
    ("carcontrol_accel", "actuator <= -0.5", -0.5),
  ):
    t = _first_crossing(rows, column, lambda v, threshold=threshold: v <= threshold)
    if t is not None:
      lines.append(f"  first {label}: {t:.2f}s")

  for output_col, label in (("carcontrol_accel", "carControl"), ("caroutput_accel", "carOutput"), ("a_ego", "aEgo")):
    lag = _best_lag(rows, "plan_a_target", output_col)
    if lag is not None:
      lag_s, corr = lag
      lines.append(f"  plan_a vs {label}: best_lag={lag_s:.2f}s corr={corr:.2f}")

  source_changes = _source_changes(rows, "plan_source")
  if source_changes:
    lines.append(f"  plan source changes: {len(source_changes)} ({', '.join(source_changes[:8])})")
  sp_reasons = _counter(rows, "sp_selected_reason")
  if sp_reasons:
    lines.append(f"  SP selected reasons: {_format_counter(sp_reasons)}")
  standstill_positive = _count_positive_accel_at_standstill(rows)
  if standstill_positive:
    lines.append(f"  positive accel at standstill samples: {standstill_positive}")

  ttc = _ttc_stats(rows)
  if ttc:
    lines.append(f"  TTC while closing: {ttc}")
  jerk = _min_derivative(rows, "plan_a_target")
  if jerk is not None:
    lines.append(f"  min target jerk proxy: {jerk:.2f} m/s^3")
  classification = _classify_longitudinal(rows)
  lines.append(f"  classification: {classification}")
  return lines


def _render_lateral(rows: list[dict[str, Any]]) -> list[str]:
  lines: list[str] = []
  for column, label in (
    ("steering_angle_deg", "steeringAngleDeg"),
    ("carcontrol_torque", "carControl.torque"),
    ("caroutput_torque", "carOutput.torque"),
    ("controls_desired_curvature", "desiredCurvature"),
    ("controls_curvature", "measuredCurvature"),
    ("model_path_raw_curvature", "rawModelCurvature"),
    ("model_path_processed_curvature", "processedCurvature"),
    ("lat_desired_lateral_accel", "desiredLatAccel"),
    ("lat_actual_lateral_accel", "actualLatAccel"),
  ):
    stat = _stats(rows, column)
    if stat:
      lines.append(f"  {label}: {stat}")

  active_ratio = _true_ratio(rows, "carcontrol_lat_active")
  override_ratio = _true_ratio(rows, "steering_pressed")
  gated_ratio = _true_ratio(rows, "model_path_gated")
  saturated_ratio = _true_ratio(rows, "lat_state_saturated")
  if active_ratio is not None:
    lines.append(f"  lat active: {active_ratio:.1f}%")
  if override_ratio is not None:
    lines.append(f"  steering override: {override_ratio:.1f}%")
  if gated_ratio is not None:
    lines.append(f"  path gated: {gated_ratio:.1f}%")
  if saturated_ratio is not None:
    lines.append(f"  lateral saturated: {saturated_ratio:.1f}%")

  desired_actual = _tracking_rms(rows, "controls_desired_curvature", "controls_curvature")
  if desired_actual is not None:
    lines.append(f"  desired-vs-measured curvature RMS: {desired_actual:.6f}")
  wrong_sign = _wrong_sign_ratio(rows, "controls_desired_curvature", "controls_curvature", eps=1e-5)
  if wrong_sign is not None:
    lines.append(f"  wrong-sign desired/measured curvature: {wrong_sign:.1f}%")
  desired_energy = _oscillation_energy(rows, "controls_desired_curvature")
  torque_energy = _oscillation_energy(rows, "carcontrol_torque")
  if desired_energy is not None:
    lines.append(f"  desired-curvature derivative energy: {desired_energy:.6g}")
  if torque_energy is not None:
    lines.append(f"  torque derivative energy: {torque_energy:.6g}")
  path_reasons = _counter(rows, "model_path_reason")
  if path_reasons:
    lines.append(f"  path reasons: {_format_counter(path_reasons)}")
  classification = _classify_lateral(rows)
  lines.append(f"  classification: {classification}")
  return lines


def _stats(rows: list[dict[str, Any]], column: str) -> str | None:
  _t, values = _series(rows, column)
  if values.size == 0:
    return None
  return f"min={np.min(values):.3f} max={np.max(values):.3f} first={values[0]:.3f} last={values[-1]:.3f}"


def _series(rows: list[dict[str, Any]], column: str) -> tuple[np.ndarray, np.ndarray]:
  t: list[float] = []
  values: list[float] = []
  last_t: float | None = None
  last_value: float | None = None
  for row in rows:
    time_s = _float_or_none(row.get("time_s"))
    value = _float_or_none(row.get(column))
    if time_s is None or value is None:
      continue
    if last_t == time_s and last_value == value:
      continue
    t.append(time_s)
    values.append(value)
    last_t = time_s
    last_value = value
  return np.array(t, dtype=float), np.array(values, dtype=float)


def _first_crossing(rows: list[dict[str, Any]], column: str, predicate: Any) -> float | None:
  for row in rows:
    value = _float_or_none(row.get(column))
    time_s = _float_or_none(row.get("time_s"))
    if value is not None and time_s is not None and predicate(value):
      return time_s
  return None


def _best_lag(rows: list[dict[str, Any]], reference_col: str, follower_col: str) -> tuple[float, float] | None:
  tref, x = _series(rows, reference_col)
  tfollow, y = _series(rows, follower_col)
  if x.size < 5 or y.size < 5:
    return None
  best: tuple[float, float] | None = None
  for lag_s in np.arange(-1.0, 3.05, 0.05):
    shifted = tref + lag_s
    mask = (shifted >= tfollow[0]) & (shifted <= tfollow[-1])
    if int(np.sum(mask)) < 5:
      continue
    yi = np.interp(shifted[mask], tfollow, y)
    xi = x[mask]
    corr = _correlation(xi, yi)
    if corr is None:
      continue
    if best is None or corr > best[1]:
      best = (float(lag_s), float(corr))
  return best


def _source_changes(rows: list[dict[str, Any]], column: str) -> list[str]:
  changes: list[str] = []
  last = None
  for row in rows:
    value = str(row.get(column) or "")
    if not value:
      continue
    if last is None:
      last = value
      continue
    if value != last:
      changes.append(f"{row.get('time_s')}s {last}->{value}")
      last = value
  return changes


def _counter(rows: list[dict[str, Any]], column: str) -> Counter[str]:
  values = [str(row.get(column) or "") for row in rows]
  return Counter(v for v in values if v and v != "unknown")


def _format_counter(counter: Counter[str]) -> str:
  return ", ".join(f"{value}={count}" for value, count in counter.most_common(6))


def _count_positive_accel_at_standstill(rows: list[dict[str, Any]]) -> int:
  count = 0
  for row in rows:
    if _truthy(row.get("standstill")):
      plan = _float_or_none(row.get("plan_a_target"))
      ctrl = _float_or_none(row.get("carcontrol_accel"))
      if (plan is not None and plan > 0.05) or (ctrl is not None and ctrl > 0.05):
        count += 1
  return count


def _ttc_stats(rows: list[dict[str, Any]]) -> str | None:
  ttc_values: list[float] = []
  for row in rows:
    status = _truthy(row.get("lead0_status"))
    d_rel = _float_or_none(row.get("lead0_d_rel"))
    v_rel = _float_or_none(row.get("lead0_v_rel"))
    if status and d_rel is not None and v_rel is not None and v_rel < -0.1:
      ttc_values.append(d_rel / -v_rel)
  if not ttc_values:
    return None
  arr = np.array(ttc_values)
  return f"min={np.min(arr):.2f}s p10={np.percentile(arr, 10):.2f}s last={arr[-1]:.2f}s"


def _min_derivative(rows: list[dict[str, Any]], column: str) -> float | None:
  t, values = _series(rows, column)
  if values.size < 3:
    return None
  dt = np.diff(t)
  dv = np.diff(values)
  mask = dt > 1e-3
  if not np.any(mask):
    return None
  return float(np.min(dv[mask] / dt[mask]))


def _classify_longitudinal(rows: list[dict[str, Any]]) -> str:
  min_target = _min_value(rows, "plan_a_target")
  min_ttc = _min_ttc(rows)
  source_flips = len(_source_changes(rows, "plan_source"))
  lead_seen = any(_truthy(row.get("lead0_status")) for row in rows)
  standstill = any(_truthy(row.get("standstill")) for row in rows)
  if standstill and _count_positive_accel_at_standstill(rows):
    return "stop-release-positive-accel"
  if lead_seen and min_target is not None and min_target <= -2.0:
    if min_ttc is not None and min_ttc < 3.0:
      return "urgent/hard lead approach"
    return "peak-braking lead approach"
  if source_flips >= 2:
    return "source instability"
  if lead_seen and min_target is not None and min_target < -0.2:
    return "routine lead approach"
  return "no strong longitudinal event"


def _classify_lateral(rows: list[dict[str, Any]]) -> str:
  override_ratio = _true_ratio(rows, "steering_pressed") or 0.0
  saturated_ratio = _true_ratio(rows, "lat_state_saturated") or 0.0
  desired_energy = _oscillation_energy(rows, "controls_desired_curvature") or 0.0
  torque_energy = _oscillation_energy(rows, "carcontrol_torque") or 0.0
  wrong_sign = _wrong_sign_ratio(rows, "controls_desired_curvature", "controls_curvature", eps=1e-5) or 0.0
  if override_ratio > 20.0:
    return "driver override / low confidence"
  if saturated_ratio > 10.0:
    return "torque/governor-limited tracking"
  if wrong_sign > 20.0:
    return "possible late correction or path/controller phase lag"
  if desired_energy > 1e-5 or torque_energy > 2.0:
    return "possible hunting/oscillation"
  return "no strong lateral event"


def _min_value(rows: list[dict[str, Any]], column: str) -> float | None:
  _t, values = _series(rows, column)
  if values.size == 0:
    return None
  return float(np.min(values))


def _min_ttc(rows: list[dict[str, Any]]) -> float | None:
  values: list[float] = []
  for row in rows:
    d_rel = _float_or_none(row.get("lead0_d_rel"))
    v_rel = _float_or_none(row.get("lead0_v_rel"))
    if d_rel is not None and v_rel is not None and v_rel < -0.1:
      values.append(d_rel / -v_rel)
  return min(values) if values else None


def _true_ratio(rows: list[dict[str, Any]], column: str) -> float | None:
  values = [_truthy_or_none(row.get(column)) for row in rows]
  values = [v for v in values if v is not None]
  if not values:
    return None
  return 100.0 * sum(1 for v in values if v) / len(values)


def _tracking_rms(rows: list[dict[str, Any]], target_col: str, actual_col: str) -> float | None:
  target_t, target = _series(rows, target_col)
  actual_t, actual = _series(rows, actual_col)
  if target.size < 5 or actual.size < 5:
    return None
  mask = (target_t >= actual_t[0]) & (target_t <= actual_t[-1])
  if int(np.sum(mask)) < 5:
    return None
  actual_interp = np.interp(target_t[mask], actual_t, actual)
  return float(np.sqrt(np.mean((target[mask] - actual_interp) ** 2)))


def _wrong_sign_ratio(rows: list[dict[str, Any]], target_col: str, actual_col: str, *, eps: float) -> float | None:
  target_t, target = _series(rows, target_col)
  actual_t, actual = _series(rows, actual_col)
  if target.size < 5 or actual.size < 5:
    return None
  mask = (target_t >= actual_t[0]) & (target_t <= actual_t[-1]) & (np.abs(target) > eps)
  if int(np.sum(mask)) < 5:
    return None
  actual_interp = np.interp(target_t[mask], actual_t, actual)
  wrong = target[mask] * actual_interp < -(eps * eps)
  return 100.0 * float(np.mean(wrong))


def _oscillation_energy(rows: list[dict[str, Any]], column: str) -> float | None:
  t, values = _series(rows, column)
  if values.size < 5:
    return None
  dt = np.diff(t)
  dv = np.diff(values)
  mask = dt > 1e-3
  if not np.any(mask):
    return None
  derivative = dv[mask] / dt[mask]
  duration = max(t[-1] - t[0], 1e-3)
  return float(np.sum(derivative ** 2 * dt[mask]) / duration)


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
  if x.size != y.size or x.size < 3:
    return None
  if np.std(x) < 1e-6 or np.std(y) < 1e-6:
    return None
  return float(np.corrcoef(x, y)[0, 1])


def _set(state: dict[str, Any], column: str, value: Any) -> None:
  if value is None:
    return
  if isinstance(value, bool):
    state[column] = int(value)
  elif _is_number(value):
    state[column] = float(value)
  else:
    state[column] = str(value)


def _safe_get(obj: Any, path: str, default: Any = None) -> Any:
  cur = obj
  for part in path.split('.'):
    try:
      cur = getattr(cur, part)
    except Exception:
      return default
  return cur


def _union_which(obj: Any) -> str:
  try:
    which = getattr(obj, "which")
    if callable(which):
      return str(which())
  except Exception:
    return ""
  return ""


def _enum(value: Any) -> str:
  if value is None:
    return ""
  try:
    return format_enum(value)
  except Exception:
    return str(value)


def _first_number(values: Any) -> float | None:
  try:
    for value in values:
      if _is_number(value):
        return float(value)
  except Exception:
    return None
  return None


def _last_number(values: Any) -> float | None:
  try:
    result = None
    for value in values:
      if _is_number(value):
        result = float(value)
    return result
  except Exception:
    return None


def _float_or_none(value: Any) -> float | None:
  try:
    if value == "":
      return None
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if isfinite(result) else None


def _truthy_or_none(value: Any) -> bool | None:
  if value == "" or value is None:
    return None
  return _truthy(value)


def _truthy(value: Any) -> bool:
  if isinstance(value, str):
    return value.lower() in {"1", "true", "yes"}
  return bool(value)


def _is_number(value: Any) -> bool:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return False
  return isfinite(result)


if __name__ == "__main__":
  main()
