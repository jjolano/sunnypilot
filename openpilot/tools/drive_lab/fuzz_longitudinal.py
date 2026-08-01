#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.tools.drive_lab.log_profile import load_profile
from openpilot.tools.drive_lab.longitudinal_scenarios import (
  LAUNCH_START_ORACLE_KINDS,
  REALISM_MODES,
  SCENARIO_PRESETS,
  PresetRequest,
  Scenario,
  generate_openpilot_acc_scenarios,
  generate_preset_scenarios,
  generate_scenarios,
  generate_udacity_acc_scenarios,
  scenario_maneuver_kwargs,
)
from openpilot.tools.drive_lab.metrics import ScenarioFailure, evaluate_maneuver_output
from openpilot.tools.drive_lab.oracle_profiles import OracleProfile, get_oracle_profile
from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec

# Re-export for backward compatibility.
__all__ = [
  "Scenario",
  "ScenarioResult",
  "REALISM_MODES",
  "SCENARIO_PRESETS",
  "MODE_DEFAULT_JERK",
  "LAUNCH_START_ORACLE_KINDS",
  "generate_scenarios",
  "generate_udacity_acc_scenarios",
  "generate_openpilot_acc_scenarios",
  "evaluate_invariants",
  "evaluate_accordion_response",
  "evaluate_lead_pullaway_start",
  "evaluate_collision_response",
  "scenario_maneuver_kwargs",
  "shipped_longitudinal_config",
  "run_scenario",
  "render_maneuver_snippet",
  "scenario_to_spec",
  "scenario_to_dict",
]

MODE_DEFAULT_JERK = {
  "comfort": 8.0,
  "emergency": 12.0,
  "adversarial": 100.0,
}

LEAD_PULLAWAY_MOVING_SPEED = 0.5
LEAD_PULLAWAY_STARTED_SPEED = 0.2
LEAD_PULLAWAY_STARTED_ACCEL = 0.1
COLLISION_GAP = 0.4
BEST_EFFORT_BRAKE = 2.5
# a single frame at the brake threshold is not a best effort; require it held
BEST_EFFORT_MIN_S = 0.5
BENIGN_IMPACT_SPEED = 3.0

# These start ego and lead at the same speed, so total speed variation measures whether
# the follower absorbs the wave or amplifies it. Cached manual stop-to-stop cycles (routes
# 288/28b/290/291/296) had median peak-speed gain 0.88 and p90 1.01; the deterministic
# gate keeps the simpler invariant: ego variation must not exceed the lead's.
ACCORDION_ORACLE_KINDS = frozenset({
  "udacity_acc_oscillating_lead",
  "udacity_acc_stop_and_go_10mph",
  "iso15622_stop_and_go",
  "iihs_stop_and_go_smooth",
})
MAX_ACCORDION_SPEED_VARIATION_GAIN = 1.0
MIN_ACCORDION_LEAD_VARIATION = 1.0

_SHIPPED_LONGITUDINAL_PARAM_KEYS = (
  "CustomLongitudinalEnabled",
  "CustomLongitudinalMode",
  "LongitudinalPersonality",
  "LongitudinalDebugTraceMode",
  "CutInBrakeAssistMode",
  "CurveTrafficAdvisorMode",
  "MapCoastMode",
  "StandstillReleaseConfidenceMode",
  "SmartCruiseControlVision",
  "SmartCruiseControlMap",
)


@dataclass(frozen=True)
class SlewCall:
  idx: int
  input_a: float
  output_a: float
  capped: bool


@dataclass(frozen=True)
class CommandFrame:
  idx: int
  time_s: float
  a_cmd: float
  a_plant: float
  v_ego: float
  v_lead: float
  d_rel: float
  prob_lead: float
  output_should_stop: bool
  debug: dict[str, Any] = field(default_factory=dict)
  custom: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JerkDiagnosis:
  idx0: int
  idx1: int
  time0: float
  time1: float
  dt: float
  jerk_window: int
  a0: float
  a1: float
  delta_a: float
  jerk: float
  frames: list[CommandFrame] = field(default_factory=list)
  slew_input: float | None = None
  slew_output: float | None = None
  slew_capped: bool | None = None


@dataclass(frozen=True)
class ScenarioResult:
  scenario: Scenario
  valid: bool
  failures: list[ScenarioFailure]
  mpc_solution_status_counts: dict[int, int] = field(default_factory=dict)
  jerk_diagnosis: JerkDiagnosis | None = None


def aggregate_mpc_solution_status_counts(results: list[ScenarioResult]) -> dict[int, int]:
  total: dict[int, int] = {}
  for result in results:
    for status, count in result.mpc_solution_status_counts.items():
      total[status] = total.get(status, 0) + count
  return total


def diagnose_max_jerk(frames: list[CommandFrame], jerk_window: int,
                      slew_calls: list[SlewCall] | None = None) -> JerkDiagnosis | None:
  if not frames or jerk_window < 1:
    return None
  window = max(1, int(jerk_window))
  n = len(frames)
  if n <= window:
    return None

  times = np.array([f.time_s for f in frames], dtype=float)
  cmds = np.array([f.a_cmd for f in frames], dtype=float)
  dt_arr = times[window:] - times[:-window]
  valid_dt = np.isfinite(dt_arr) & (dt_arr > 1e-6)
  delta_arr = cmds[window:] - cmds[:-window]
  valid = valid_dt & np.isfinite(delta_arr) & _same_authority_mask(
    [f.output_should_stop for f in frames], n, window,
  )
  if not np.any(valid):
    return None

  jerk_arr = np.full(delta_arr.shape, np.nan, dtype=float)
  jerk_arr[valid] = delta_arr[valid] / dt_arr[valid]
  max_idx = int(np.nanargmax(np.abs(jerk_arr)))
  idx0 = max_idx
  idx1 = max_idx + window

  a0 = float(cmds[idx0])
  a1 = float(cmds[idx1])
  dt = float(times[idx1] - times[idx0])
  delta_a = a1 - a0
  jerk = delta_a / dt if dt > 1e-6 else 0.0

  window_frames = list(frames[idx0:idx1 + 1])

  slew_input: float | None = None
  slew_output: float | None = None
  slew_capped: bool | None = None
  calls = slew_calls or []
  if calls:
    window_calls = [c for c in calls if idx0 <= c.idx <= idx1]
    call = next((c for c in window_calls if c.idx == idx1), window_calls[-1] if window_calls else None)
    if call is not None:
      slew_input, slew_output, slew_capped = call.input_a, call.output_a, call.capped

  return JerkDiagnosis(
    idx0=idx0,
    idx1=idx1,
    time0=float(times[idx0]),
    time1=float(times[idx1]),
    dt=dt,
    jerk_window=window,
    a0=a0,
    a1=a1,
    delta_a=delta_a,
    jerk=jerk,
    frames=window_frames,
    slew_input=slew_input,
    slew_output=slew_output,
    slew_capped=slew_capped,
  )


def _same_authority_mask(should_stop: Any, frame_count: int, window: int) -> np.ndarray:
  flags = np.asarray(should_stop, dtype=bool)
  if flags.shape != (frame_count,):
    return np.ones(frame_count - window, dtype=bool)
  return ~flags[:-window] & ~flags[window:]


def _max_eligible_jerk(time_s: np.ndarray, accel: np.ndarray, should_stop: Any,
                       window: int) -> float | None:
  if len(time_s) <= window + 1 or accel.shape != time_s.shape:
    return None
  dt = time_s[window:] - time_s[:-window]
  delta = accel[window:] - accel[:-window]
  valid = (
    np.isfinite(dt) & (dt > 1e-6) & np.isfinite(delta) &
    _same_authority_mask(should_stop, len(time_s), window)
  )
  if not np.any(valid):
    return None
  return float(np.max(np.abs(delta[valid] / dt[valid])))


def _frame_release_gate_context(frame: CommandFrame, prev_frame: CommandFrame | None,
                                stopping_distance: float | None = None) -> dict[str, Any]:
  if stopping_distance is None:
    from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
    stopping_distance = STOP_DISTANCE

  custom = frame.custom
  debug = frame.debug
  d_rel = frame.d_rel
  v_lead = frame.v_lead
  v_ego = frame.v_ego
  v_rel = v_lead - v_ego
  lead_id = custom.get("selected_lead_id")
  latch_id = custom.get("lead_stop_hold_lead_id")
  effective_latch_id = latch_id if latch_id is not None else custom.get("lead_stop_hold_lead_id_before_reset")
  same_id = lead_id is not None and effective_latch_id is not None and lead_id == effective_latch_id
  baseline = custom.get("lead_stop_hold_gap_baseline_d_rel")
  if baseline is None:
    baseline = custom.get("lead_stop_hold_gap_baseline_d_rel_before_reset")
  prev_d_rel = custom.get("lead_stop_hold_gap_prev_d_rel")
  if prev_d_rel is None:
    prev_d_rel = custom.get("lead_stop_hold_gap_prev_d_rel_before_reset")
  gap_increasing_s = float(custom.get("lead_stop_hold_gap_increasing_s_before_reset", custom.get("lead_stop_hold_gap_increasing_s", 0.0)))
  current_active = bool(custom.get("lead_stop_hold_active", False))
  prev_active = bool(prev_frame.custom.get("lead_stop_hold_active", False)) if prev_frame is not None else False
  latch_reset = bool(custom.get("lead_stop_hold_reset_on_frame", False)) or (prev_active and not current_active)

  release_min = stopping_distance + (0.2 if same_id else 0.1)
  baseline_opening: float | None = None
  if same_id and baseline is not None and math.isfinite(float(baseline)):
    baseline = float(baseline)
    baseline_opening = d_rel - baseline
    baseline_min_d_rel = baseline + 0.5
    release_min = max(4.5, min(release_min, baseline_min_d_rel))

  prep_min = stopping_distance + 0.20

  mpc_a = float(debug.get("mpc_a_target", 0.0)) if debug else 0.0
  model_a = float(debug.get("model_a_target", 0.0)) if debug else 0.0
  model_stop = bool(debug.get("model_should_stop", False)) if debug else False

  release_source = str(custom.get("standstill_release_source", ""))
  release_allowed = bool(custom.get("standstill_release_allowed", False))
  output_should_stop = bool(debug.get("final_should_stop", frame.output_should_stop))

  release_path = "normal_mpc"
  if current_active:
    release_path = "lead_stop_hold_active"
  elif latch_reset:
    release_path = "lead_stop_hold_release"
  elif release_allowed and release_source in ("lead_pullaway", "lead_standstill_launch", "no_lead_launch"):
    release_path = "standstill_release_clear"
  elif not output_should_stop:
    release_path = "normal_mpc"

  prep_block_reason = "different_or_missing_lead_id"
  prep_applies = False
  prep_gate_would_apply = False
  if not custom.get("custom_output_enabled", False) or not custom.get("custom_long_enabled", False):
    prep_block_reason = "custom_output_unavailable"
  elif not release_allowed:
    prep_block_reason = "no_release_permission"
  elif release_source not in ("lead_pullaway", "lead_standstill_launch"):
    prep_block_reason = "invalid_release_source"
  elif custom.get("custom_should_stop", False):
    prep_block_reason = "custom_should_stop"
  elif model_stop:
    prep_block_reason = "raw_model_stop"
  elif custom.get("driver_brake", False):
    prep_block_reason = "driver_brake"
  elif custom.get("driver_gas", False):
    prep_block_reason = "driver_gas"
  elif custom.get("force_decel", False):
    prep_block_reason = "force_decel"
  elif v_ego >= 0.7:
    prep_block_reason = "not_near_standstill"
  elif lead_id is None or effective_latch_id is None or lead_id != effective_latch_id:
    prep_block_reason = "different_or_missing_lead_id"
  elif not all(math.isfinite(x) for x in (d_rel, v_lead, v_rel, mpc_a, model_a)):
    prep_block_reason = "non_finite_values"
  elif mpc_a < -0.10:
    prep_block_reason = "mpc_brake_veto"
  elif v_lead < 0.25 or v_rel < 0.10:
    prep_block_reason = "lead_not_moving"
  elif gap_increasing_s < 0.15:
    prep_block_reason = "gap_increasing_time"
  elif d_rel <= prep_min:
    prep_block_reason = "distance_gate"
  else:
    prep_gate_would_apply = True
    if current_active:
      prep_block_reason = "applies"
      prep_applies = True
    else:
      prep_block_reason = "not_hold_branch"

  return {
    "selected_lead_id": lead_id,
    "lead_stop_hold_lead_id": latch_id,
    "effective_lead_stop_hold_lead_id": effective_latch_id,
    "lead_stop_hold_gap_increasing_s": gap_increasing_s,
    "lead_stop_hold_gap_baseline_d_rel": baseline,
    "lead_stop_hold_gap_prev_d_rel": prev_d_rel,
    "baseline_opening": baseline_opening,
    "release_min_d_rel": release_min,
    "prep_min_d_rel": prep_min,
    "d_rel_minus_release_min_d_rel": d_rel - release_min,
    "d_rel_minus_prep_min_d_rel": d_rel - prep_min,
    "prep_applies": prep_applies,
    "prep_gate_would_apply": prep_gate_would_apply,
    "prep_block_reason": prep_block_reason,
    "release_path": release_path,
    "latch_reset_on_frame": latch_reset,
    "same_id": same_id,
  }


def evaluate_invariants(
  valid: bool,
  output: np.ndarray,
  max_normal_jerk: float = 8.0,
  commanded_accel: np.ndarray | None = None,
  jerk_window: int = 1,
  *,
  should_stop: np.ndarray | None = None,
  profile: OracleProfile | None = None,
) -> list[ScenarioFailure]:
  oracle = profile or get_oracle_profile("comfort")
  if oracle.skip_jerk:
    max_normal_jerk = oracle.max_jerk_override or max_normal_jerk
  elif oracle.max_jerk_override is not None:
    max_normal_jerk = oracle.max_jerk_override
  result = evaluate_maneuver_output("legacy", valid, output, max_normal_jerk, commanded_accel, jerk_window)
  failures = list(result.failures)
  if should_stop is not None and output.ndim == 2 and output.shape[1] >= 7 and output.size:
    accel = np.asarray(commanded_accel, dtype=float) if commanded_accel is not None else output[:, 5]
    authority = np.asarray(should_stop, dtype=bool)
    if authority.shape == output[:, 0].shape and accel.shape == output[:, 0].shape:
      failures = [failure for failure in failures if failure.check != "jerk"]
    if (
      authority.shape == output[:, 0].shape and accel.shape == output[:, 0].shape and
      "jerk" in oracle.checks and not oracle.skip_jerk and np.all(np.isfinite(output))
    ):
      max_jerk = _max_eligible_jerk(output[:, 0], accel, should_stop, max(1, int(jerk_window)))
      if max_jerk is not None and max_jerk > max_normal_jerk:
        failures.append(ScenarioFailure("jerk", f"maximum absolute jerk {max_jerk:.3f} m/s^3"))
  if oracle.skip_jerk:
    failures = [f for f in failures if f.check != "jerk"]
  if "valid" not in oracle.checks:
    failures = [f for f in failures if f.check != "valid"]
  if "collision" not in oracle.checks:
    failures = [f for f in failures if f.check != "collision"]
  if "speed" not in oracle.checks:
    failures = [f for f in failures if f.check != "speed"]
  return failures


def evaluate_accordion_response(output: np.ndarray) -> list[ScenarioFailure]:
  """Fail when ego reproduces more speed variation than the lead."""
  if output.ndim != 2 or output.shape[1] < 5 or output.size == 0:
    return []

  v_ego = output[:, 3]
  v_lead = output[:, 4]
  if not (np.all(np.isfinite(v_ego)) and np.all(np.isfinite(v_lead))):
    return []  # the generic finite-output oracle owns malformed trajectories

  lead_variation = float(np.sum(np.abs(np.diff(v_lead))))
  if lead_variation < MIN_ACCORDION_LEAD_VARIATION:
    return []
  ego_variation = float(np.sum(np.abs(np.diff(v_ego))))
  gain = ego_variation / lead_variation
  if gain <= MAX_ACCORDION_SPEED_VARIATION_GAIN + 1e-6:
    return []
  return [ScenarioFailure(
    "accordion",
    f"ego speed variation {ego_variation:.3f} m/s amplifies lead variation "
    f"{lead_variation:.3f} m/s (gain {gain:.3f})",
  )]


@dataclass
class CommandCapture:
  commanded: list[float] = field(default_factory=list)
  prob_lead: list[float] = field(default_factory=list)
  mpc_solution_status_counts: dict[int, int] = field(default_factory=dict)
  frames: list[CommandFrame] = field(default_factory=list)
  slew_calls: list[SlewCall] = field(default_factory=list)
  lead_stop_hold_reset_contexts: dict[int, dict[str, Any]] = field(default_factory=dict)
  current_frame_idx: int = 0


@contextlib.contextmanager
def capture_commanded_accel():
  from openpilot.selfdrive.test.longitudinal_maneuvers import plant as plant_module
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, STOP_DISTANCE
  Plant = plant_module.Plant

  capture = CommandCapture()
  original_step = Plant.step
  original_sleep = plant_module.time.sleep
  original_reset = LongitudinalMpc.reset

  def step(self, *step_args, **step_kwargs):
    capture.current_frame_idx = len(capture.frames)
    result = original_step(self, *step_args, **step_kwargs)
    idx = capture.current_frame_idx
    a_cmd = float(self.planner.output_a_target)
    output_should_stop = bool(self.planner.output_should_stop)
    prob_lead = step_kwargs.get("prob_lead", step_args[1] if len(step_args) > 1 else 1.0)
    prob_lead = float(prob_lead)
    capture.commanded.append(a_cmd)
    capture.prob_lead.append(prob_lead)

    v_lead = float(step_kwargs.get("v_lead", step_args[0] if step_args else 0.0))
    a_plant = float(self.acceleration)
    v_ego = float(self.speed)
    if getattr(self, "lead_relevancy", False):
      d_rel = max(0.0, float(self.distance_lead - self.distance))
    else:
      d_rel = 200.0

    debug = dict(getattr(self.planner, "_last_longitudinal_debug", {}) or {})
    custom: dict[str, Any] = {}
    custom_output = getattr(self.planner, "custom_long_output", None)
    if custom_output is not None:
      custom["selected_intent"] = str(getattr(custom_output, "selected_intent", "") or "")
      custom["reason"] = str(getattr(custom_output, "reason", "") or "")
      custom["standstill_release_allowed"] = bool(getattr(custom_output, "standstill_release_allowed", False))
      custom["standstill_release_source"] = str(getattr(custom_output, "standstill_release_source", "") or "")
      custom["standstill_release_a_target"] = float(getattr(custom_output, "standstill_release_a_target", 0.0))
      custom["custom_should_stop"] = bool(getattr(custom_output, "should_stop", False))
      custom["custom_output_enabled"] = bool(getattr(custom_output, "enabled", False))
    custom_long = getattr(self.planner, "custom_long", None)
    custom["custom_long_enabled"] = bool(custom_long is not None and getattr(custom_long, "enabled", False))
    custom["custom_long_mode"] = str(getattr(custom_long, "mode", ""))
    custom["standstill_release_confidence_mode"] = str(getattr(custom_long, "standstill_release_confidence_mode", "off"))
    custom["driver_brake"] = False
    custom["driver_gas"] = False
    custom["force_decel"] = bool(getattr(self, "force_decel", False))
    custom["release_block_reason"] = str(getattr(self.planner, "_last_release_block_reason", "") or "")
    custom["lead_stop_hold_active"] = bool(getattr(self.planner, "_lead_stop_hold_active", False))
    custom["lead_stop_hold_lead_id"] = getattr(self.planner, "_lead_stop_hold_lead_id", None)
    lead_active = bool(getattr(self, "lead_relevancy", False) and (float(prob_lead) > 0.5 or getattr(self, "only_radar", False)))
    custom["selected_lead_id"] = 1 if lead_active else None
    custom["lead_stop_hold_gap_increasing_s"] = float(getattr(self.planner, "_lead_stop_hold_gap_increasing_s", 0.0))
    gap_baseline = getattr(self.planner, "_lead_stop_hold_gap_baseline_d_rel", None)
    custom["lead_stop_hold_gap_baseline_d_rel"] = gap_baseline
    custom["lead_stop_hold_gap_prev_d_rel"] = getattr(self.planner, "_lead_stop_hold_gap_prev_d_rel", None)
    reset_context = capture.lead_stop_hold_reset_contexts.get(idx)
    if reset_context is not None:
      custom.update(reset_context)
      custom["lead_stop_hold_reset_on_frame"] = True
    else:
      custom["lead_stop_hold_reset_on_frame"] = False
    custom["stop_hold_release_slew_a_target"] = getattr(self.planner, "_stop_hold_release_slew_a_target", None)

    frame = CommandFrame(
      idx=idx,
      time_s=float(self.current_time),
      a_cmd=a_cmd,
      a_plant=a_plant,
      v_ego=v_ego,
      v_lead=v_lead,
      d_rel=d_rel,
      prob_lead=prob_lead,
      output_should_stop=output_should_stop,
      debug=debug,
      custom=custom,
    )
    prev_frame = capture.frames[-1] if capture.frames else None
    stopping_distance = float(getattr(self.planner.CP, 'stoppingDistance', STOP_DISTANCE) or STOP_DISTANCE)
    frame.custom.update(_frame_release_gate_context(frame, prev_frame, stopping_distance))
    capture.frames.append(frame)
    return result

  def reset(self, *reset_args, **reset_kwargs):
    status = getattr(self, "solution_status", 0)
    if isinstance(status, (int, np.integer)) and status != 0:
      capture.mpc_solution_status_counts[int(status)] = capture.mpc_solution_status_counts.get(int(status), 0) + 1
    return original_reset(self, *reset_args, **reset_kwargs)

  apply_slew_module = None
  try:
    from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
    apply_slew_module = LongitudinalPlannerSP
  except Exception:
    pass

  original_apply_slew = None
  original_reset_lead_stop_hold = None
  if apply_slew_module is not None and hasattr(apply_slew_module, "_apply_stop_hold_release_slew"):
    original_apply_slew = apply_slew_module._apply_stop_hold_release_slew

    def apply_stop_hold_release_slew(self, sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop):
      input_a = float(a_target)
      output_a = original_apply_slew(self, sm, a_target, release_mpc_stop, mpc_stop, raw_model_should_stop, should_stop)
      output_a = float(output_a)
      capped = bool(math.isfinite(input_a) and math.isfinite(output_a) and abs(output_a - input_a) > 1e-6)
      capture.slew_calls.append(SlewCall(capture.current_frame_idx, input_a, output_a, capped))
      return output_a

    apply_slew_module._apply_stop_hold_release_slew = apply_stop_hold_release_slew

  if apply_slew_module is not None and hasattr(apply_slew_module, "_reset_lead_stop_hold"):
    original_reset_lead_stop_hold = apply_slew_module._reset_lead_stop_hold

    def reset_lead_stop_hold(self):
      idx = capture.current_frame_idx
      capture.lead_stop_hold_reset_contexts[idx] = {
        "lead_stop_hold_active_before_reset": bool(getattr(self, "_lead_stop_hold_active", False)),
        "lead_stop_hold_lead_id_before_reset": getattr(self, "_lead_stop_hold_lead_id", None),
        "lead_stop_hold_gap_increasing_s_before_reset": float(getattr(self, "_lead_stop_hold_gap_increasing_s", 0.0)),
        "lead_stop_hold_gap_baseline_d_rel_before_reset": getattr(self, "_lead_stop_hold_gap_baseline_d_rel", None),
        "lead_stop_hold_gap_prev_d_rel_before_reset": getattr(self, "_lead_stop_hold_gap_prev_d_rel", None),
      }
      return original_reset_lead_stop_hold(self)

    apply_slew_module._reset_lead_stop_hold = reset_lead_stop_hold

  Plant.step = step
  plant_module.time.sleep = lambda *a, **k: None
  LongitudinalMpc.reset = reset
  try:
    yield capture
  finally:
    Plant.step = original_step
    plant_module.time.sleep = original_sleep
    LongitudinalMpc.reset = original_reset
    if original_apply_slew is not None and apply_slew_module is not None:
      apply_slew_module._apply_stop_hold_release_slew = original_apply_slew
    if original_reset_lead_stop_hold is not None and apply_slew_module is not None:
      apply_slew_module._reset_lead_stop_hold = original_reset_lead_stop_hold


def evaluate_lead_pullaway_start(output: np.ndarray) -> list[ScenarioFailure]:
  if output.ndim != 2 or output.shape[1] < 6 or output.size == 0:
    return []

  time_s = output[:, 0]
  speed = output[:, 3]
  lead_speed = output[:, 4]
  accel = output[:, 5]
  lead_moving = lead_speed > LEAD_PULLAWAY_MOVING_SPEED
  if not np.any(lead_moving):
    return []

  lead_move_time = float(time_s[int(np.flatnonzero(lead_moving)[0])])
  after_lead_moves = time_s >= lead_move_time
  started = np.any(speed[after_lead_moves] > LEAD_PULLAWAY_STARTED_SPEED) or np.any(accel[after_lead_moves] > LEAD_PULLAWAY_STARTED_ACCEL)
  if started:
    return []
  return [ScenarioFailure("launch", "lead moved but ego never started")]


def evaluate_collision_response(
  output: np.ndarray,
  commanded_accel: np.ndarray | None,
  prob_lead: np.ndarray | None,
  *,
  max_impact_speed_ms: float | None = None,
  use_best_effort: bool = True,
) -> list[ScenarioFailure]:
  if output.ndim != 2 or output.shape[1] < 7 or output.size == 0:
    return []

  v_ego = output[:, 3]
  v_lead = output[:, 4]
  d_rel = output[:, 6]

  finite = np.isfinite(d_rel)
  contact = np.flatnonzero(finite & (d_rel < COLLISION_GAP))
  if contact.size == 0:
    return []
  impact = int(contact[0])
  min_gap = float(np.min(d_rel[finite]))
  impact_speed = max(0.0, float(v_ego[impact] - v_lead[impact]))

  if max_impact_speed_ms is not None and impact_speed > max_impact_speed_ms:
    return [ScenarioFailure("collision", f"impact speed {impact_speed:.1f} m/s exceeds limit {max_impact_speed_ms:.1f} m/s")]

  if not use_best_effort:
    return [ScenarioFailure("collision", f"contact at {impact_speed:.1f} m/s (minimum lead gap {min_gap:.3f} m)")]

  detected = np.flatnonzero(np.asarray(prob_lead) > 0.5) if prob_lead is not None else np.empty(0, dtype=int)
  d0 = min(int(detected[0]) if detected.size else 0, impact)

  # "Best effort" must mean the planner actually committed to braking, not that a
  # single frame dipped below the threshold. One -2.5 m/s^2 sample anywhere in the
  # approach used to excuse an arbitrarily hard collision; require the brake to be
  # held for BEST_EFFORT_MIN_S instead.
  best_effort = False
  if commanded_accel is not None and len(commanded_accel) == len(output):
    braking = np.asarray(commanded_accel[d0:impact + 1]) <= -BEST_EFFORT_BRAKE
    if braking.any():
      # longest consecutive run of committed braking
      edges = np.diff(np.concatenate(([0], braking.view(np.int8), [0])))
      starts = np.flatnonzero(edges == 1)
      ends = np.flatnonzero(edges == -1)
      longest = int((ends - starts).max()) if starts.size else 0
      best_effort = longest * DT_MDL >= BEST_EFFORT_MIN_S
  if best_effort or impact_speed <= BENIGN_IMPACT_SPEED:
    return []
  return [
    ScenarioFailure("collision", f"hard collision at {impact_speed:.1f} m/s without full braking (minimum lead gap {min_gap:.3f} m)")
  ]


@contextlib.contextmanager
def shipped_longitudinal_config():
  from openpilot.common.params import Params

  params = Params()
  previous = {key: params.get(key) for key in _SHIPPED_LONGITUDINAL_PARAM_KEYS}
  for key in _SHIPPED_LONGITUDINAL_PARAM_KEYS:
    default = params.get_default_value(key)
    if default is None:
      params.remove(key)
    else:
      params.put(key, default, block=True)
  try:
    yield
  finally:
    for key, value in previous.items():
      if value is None:
        params.remove(key)
      else:
        params.put(key, value, block=True)


def run_scenario(scenario: Scenario, max_normal_jerk: float = 8.0) -> ScenarioResult:
  from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver

  profile = get_oracle_profile(scenario.oracle_profile)
  if profile.max_jerk_override is not None:
    max_normal_jerk = profile.max_jerk_override

  maneuver = Maneuver(scenario.title, scenario.duration, **scenario_maneuver_kwargs(scenario))
  with contextlib.redirect_stdout(io.StringIO()), capture_commanded_accel() as capture:
    valid, output = maneuver.evaluate()
  commanded_accel = np.array(capture.commanded) if len(capture.commanded) == len(output) else None
  should_stop = np.array([frame.output_should_stop for frame in capture.frames]) if len(capture.frames) == len(output) else None
  jerk_window = 1
  failures = evaluate_invariants(
    valid, output, max_normal_jerk, commanded_accel, jerk_window,
    should_stop=should_stop, profile=profile,
  )

  if scenario.kind in ACCORDION_ORACLE_KINDS:
    failures.extend(evaluate_accordion_response(output))

  if profile.use_launch_oracle and scenario.kind in LAUNCH_START_ORACLE_KINDS:
    failures.extend(evaluate_lead_pullaway_start(output))

  prob_lead = np.array(capture.prob_lead) if len(capture.prob_lead) == len(output) else None
  oracle = evaluate_collision_response(
    output,
    commanded_accel,
    prob_lead,
    max_impact_speed_ms=profile.max_impact_speed_ms,
    use_best_effort=profile.use_best_effort_collision,
  )
  contact = output.ndim == 2 and output.shape[1] >= 7 and bool(np.any(np.isfinite(output[:, 6]) & (output[:, 6] < COLLISION_GAP)))
  if contact and "collision" in profile.checks:
    if profile.use_best_effort_collision:
      failures = [f for f in failures if f.check not in ("collision", "valid")] + oracle
    else:
      failures = [f for f in failures if f.check != "collision"] + oracle

  jerk_diagnosis: JerkDiagnosis | None = None
  if any(f.check == "jerk" for f in failures):
    jerk_diagnosis = diagnose_max_jerk(capture.frames, jerk_window, capture.slew_calls)

  return ScenarioResult(scenario, not failures, failures, capture.mpc_solution_status_counts, jerk_diagnosis)


def render_maneuver_snippet(scenario: Scenario) -> str:
  kwargs = ",\n".join(f"    {key}={repr(value)}" for key, value in scenario.kwargs.items())
  return f"# mode: {scenario.mode} oracle: {scenario.oracle_profile}\nManeuver(\n    {scenario.title!r},\n    duration={scenario.duration!r},\n{kwargs}\n)"


def scenario_to_spec(scenario: Scenario, source: str = "generated", seed: int | None = None, index: int | None = None) -> ScenarioSpec:
  provenance = dict(scenario.provenance or {})
  return ScenarioSpec.from_maneuver_kwargs(
    kind=scenario.kind,
    title=scenario.title,
    mode=scenario.mode,
    duration=scenario.duration,
    kwargs=scenario.kwargs,
    source=source,
    seed=seed,
    index=index,
    provenance=provenance,
  )


def scenario_to_dict(scenario: Scenario, source: str | None = None, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
  payload = {
    "mode": scenario.mode,
    "kind": scenario.kind,
    "title": scenario.title,
    "duration": scenario.duration,
    "kwargs": scenario.kwargs,
    "oracleProfile": scenario.oracle_profile,
  }
  if scenario.provenance:
    payload["provenance"] = scenario.provenance
  if source is not None:
    spec = scenario_to_spec(scenario, source=source, seed=seed, index=index)
    payload["scenarioId"] = spec.scenario_id
    payload["spec"] = spec.to_dict()
  return payload


def render_jerk_diagnosis(d: JerkDiagnosis) -> str:
  frame = next((f for f in d.frames if f.custom.get("latch_reset_on_frame", False)), d.frames[-1] if d.frames else None)
  first_frame = d.frames[0] if d.frames else None
  last_frame = d.frames[-1] if d.frames else None
  custom = frame.custom if frame else {}
  first_should_stop = first_frame.output_should_stop if first_frame is not None else False
  last_should_stop = last_frame.output_should_stop if last_frame is not None else False
  should_stop_text = str(last_should_stop) if first_should_stop == last_should_stop else f"{first_should_stop}->{last_should_stop}"
  first_hold = first_frame.custom.get("lead_stop_hold_active", False) if first_frame is not None else False
  last_hold = last_frame.custom.get("lead_stop_hold_active", False) if last_frame is not None else False
  hold_text = str(last_hold) if first_hold == last_hold else f"{first_hold}->{last_hold}"
  release_parts = [
    f"intent={custom.get('selected_intent', '-')}",
    f"src={custom.get('standstill_release_source', '-')}",
    f"allowed={custom.get('standstill_release_allowed', False)}",
  ]
  release_block = custom.get("release_block_reason", "")
  if release_block:
    release_parts.append(f"block={release_block}")
  slew_state = custom.get("stop_hold_release_slew_a_target")
  if slew_state is not None:
    release_parts.append(f"slew_state={slew_state:.3f}")
  slew_text = "none"
  if d.slew_capped is not None:
    slew_text = f"{d.slew_input:.3f}->{d.slew_output:.3f} capped={d.slew_capped}"

  gate_parts = []
  if "prep_applies" in custom or "prep_block_reason" in custom:
    gate_parts.append(f"prep={custom.get('prep_applies', False)}")
    if custom.get("prep_gate_would_apply", False):
      gate_parts.append("prepWouldApply")
    gate_parts.append(f"prepBlock={custom.get('prep_block_reason', '-')}")
  if frame is not None:
    gate_parts.append(f"dRel={frame.d_rel:.2f}")
  if custom.get("prep_min_d_rel") is not None:
    gate_parts.append(f"prepMin={custom['prep_min_d_rel']:.2f}")
  gate_parts.append(f"releaseMin={custom.get('release_min_d_rel', 0.0):.2f}")
  gate_parts.append(f"gapS={custom.get('lead_stop_hold_gap_increasing_s', 0.0):.2f}")
  gate_parts.append(f"leadId={custom.get('selected_lead_id', '-')}")
  gate_parts.append(f"latchId={custom.get('effective_lead_stop_hold_lead_id', custom.get('lead_stop_hold_lead_id', '-'))}")
  gate_parts.append(f"path={custom.get('release_path', '-')}")
  if custom.get("latch_reset_on_frame"):
    gate_parts.append("latchReset")

  accel_text = f"raw={d.a0:.3f}->{d.a1:.3f} delta={d.delta_a:.3f} jerk={d.jerk:.3f}"

  return (
    f"jerk diagnosis: t={d.time0:.2f}->{d.time1:.2f} {accel_text} "
    f"v={frame.v_ego if frame else 0.0:.2f} "
    f"lead={frame.v_lead if frame else 0.0:.2f}@{frame.d_rel if frame else 0.0:.2f} "
    f"should_stop={should_stop_text} "
    f"release={' '.join(release_parts)} "
    f"lead_stop_hold={hold_text} "
    f"slew={slew_text} "
    f"gate={' '.join(gate_parts)}"
  )


def _snake_to_camel(s: str) -> str:
  parts = s.split('_')
  return parts[0] + ''.join(part.capitalize() for part in parts[1:])


def _to_camel_dict(obj: Any) -> Any:
  if isinstance(obj, dict):
    return {_snake_to_camel(k): _to_camel_dict(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_to_camel_dict(v) for v in obj]
  return obj


def _jerk_diagnosis_to_dict(d: JerkDiagnosis | None) -> dict[str, Any] | None:
  if d is None:
    return None
  return _to_camel_dict(asdict(d))


def main() -> None:
  parser = argparse.ArgumentParser(description="Seeded longitudinal maneuver fuzzer.")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--cases", type=int, default=100)
  parser.add_argument("--mode", choices=REALISM_MODES, default="comfort", help="Scenario realism profile")
  parser.add_argument(
    "--preset",
    choices=SCENARIO_PRESETS,
    default="fuzz",
    help="Scenario source: fuzz, udacity-acc, openpilot-acc, ncap-acc, commonroad-acc, or nuscenes-acc",
  )
  parser.add_argument("--profile", help="Optional JSON profile from profile_route.py or openacc_segments.py")
  parser.add_argument("--e2e", action="store_true", help="openpilot-acc: run maneuvers in e2e mode")
  parser.add_argument("--force-decel", action="store_true", help="openpilot-acc: enable force decel maneuvers")
  parser.add_argument("--ncap-family", choices=("CCRs", "CCRm", "CCRb"), help="ncap-acc: sample from one family")
  parser.add_argument("--ncap-sample", type=int, help="ncap-acc: number of grid points to sample")
  parser.add_argument("--commonroad-scenario", help="commonroad-acc: import a CommonRoad XML scenario")
  parser.add_argument("--nuscenes-scenario", help="nuscenes-acc: path to a nuScenes JSON export")
  parser.add_argument("--max-normal-jerk", type=float, help="Override the mode's jerk threshold")
  parser.add_argument("--max-failures", type=int, default=10)
  parser.add_argument("--list-only", action="store_true", help="Print generated scenarios without running the simulator")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
  args = parser.parse_args()

  profile = load_profile(args.profile) if args.profile else None
  request = PresetRequest(
    preset=args.preset,
    mode=args.mode,
    seed=args.seed,
    cases=args.cases,
    profile=profile,
    e2e=args.e2e,
    force_decel=args.force_decel,
    ncap_family=args.ncap_family,
    ncap_sample=args.ncap_sample,
    commonroad_scenario=args.commonroad_scenario,
    nuscenes_scenario=args.nuscenes_scenario,
  )
  scenarios = generate_preset_scenarios(request)
  if args.list_only:
    payload = [
      scenario_to_dict(
        scenario,
        source=args.preset,
        seed=args.seed if args.preset == "fuzz" else None,
        index=idx,
      )
      for idx, scenario in enumerate(scenarios)
    ]
    print(json.dumps(payload, indent=2) if args.json else "\n\n".join(render_maneuver_snippet(s) for s in scenarios))
    return

  max_normal_jerk = args.max_normal_jerk if args.max_normal_jerk is not None else MODE_DEFAULT_JERK[args.mode]
  with shipped_longitudinal_config():
    results = [run_scenario(s, max_normal_jerk) for s in scenarios]
  failures = [r for r in results if r.failures]
  mpc_counts = aggregate_mpc_solution_status_counts(results)
  if args.json:
    print(json.dumps({
      "seed": args.seed,
      "cases": args.cases,
      "mode": args.mode,
      "preset": args.preset,
      "profile": profile.source if profile is not None else None,
      "maxNormalJerk": max_normal_jerk,
      "mpcSolutionStatusCounts": dict(mpc_counts),
      "totalMpcResets": sum(mpc_counts.values()),
      "scenarioResults": [
        {
          "title": result.scenario.title,
          "kind": result.scenario.kind,
          "mpcSolutionStatusCounts": dict(result.mpc_solution_status_counts),
        }
        for result in results
      ],
      "failures": [
        {
          "scenario": scenario_to_dict(result.scenario),
          "checks": [failure.__dict__ for failure in result.failures],
          "mpcSolutionStatusCounts": dict(result.mpc_solution_status_counts),
          "jerkDiagnosis": _jerk_diagnosis_to_dict(result.jerk_diagnosis),
        }
        for result in failures
      ],
    }, indent=2))
  else:
    profile_text = f" profile={profile.source}" if profile is not None else ""
    mpc_resets_total = sum(mpc_counts.values())
    mpc_text = f" mpc_resets={mpc_resets_total}"
    if mpc_counts:
      mpc_text += f" mpc_statuses={mpc_counts}"
    print(
      f"Drive Lab fuzz preset={args.preset} seed={args.seed} mode={args.mode}{profile_text} "
      f"cases={len(scenarios)} max_normal_jerk={max_normal_jerk:g} failures={len(failures)}{mpc_text}"
    )
    for result in failures[:args.max_failures]:
      print(f"\nFAILED: {result.scenario.title} [{result.scenario.mode}/{result.scenario.kind}]")
      for failure in result.failures:
        print(f"  {failure.check}: {failure.detail}")
        if failure.check == "jerk" and result.jerk_diagnosis is not None:
          print(f"    {render_jerk_diagnosis(result.jerk_diagnosis)}")
      print(render_maneuver_snippet(result.scenario))
    mpc_results = [r for r in results if r.mpc_solution_status_counts]
    for result in mpc_results[:args.max_failures]:
      print(f"\nMPC RESET: {result.scenario.title} [{result.scenario.mode}/{result.scenario.kind}]: {result.mpc_solution_status_counts}")

  if failures:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
