#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable

from openpilot.cereal import messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.tools.drive_lab.planner_target_analysis import (
  DEFAULT_EPISODE_CONTEXT_S,
  DEFAULT_EPISODE_GAP_S,
  DEFAULT_HIGH_JERK_THRESHOLD,
  DEFAULT_LARGE_ERROR_THRESHOLD,
  MOVING_SPEED,
  PlannerTargetEpisode,
  PlannerTargetSample,
  UNSET_CRUISE_KPH,
  build_suspicious_episodes,
  high_plan_jerk_pairs,
  is_opposite_intent,
  is_strong_opposite_intent,
)
from openpilot.tools.drive_lab.route_analysis import (
  correlation as _correlation,
  finite_or_none as _finite_or_none,
  format_counts as _format_counts,
  format_optional as _format_optional,
  iter_route_messages,
  mean as _mean,
  percentile as _percentile,
  ratio as _ratio,
  route_duration,
  route_identity,
)
from openpilot.tools.drive_lab.timeline import format_enum, safe_get
from openpilot.tools.lib.logreader import LogReader, ReadMode


DEFAULT_MIN_INFERRED_CRUISE_KPH = 40.0
GPS_SERVICES = ("gpsLocation", "gpsLocationExternal")
REQUIRED_SHADOW_SERVICES = (
  "carControl",
  "carState",
  "controlsState",
  "selfdriveState",
  "liveParameters",
  "radarState",
  "modelV2",
)
OPTIONAL_SHADOW_SERVICES = ("carStateSP", "liveMapDataSP", *GPS_SERVICES)
STACK_ALIASES = {
  "sunnypilotCurrent": "sunnypilot-current",
  "sunnypilot-current": "sunnypilot-current",
  "customRecommended": "custom-recommended",
  "custom-recommended": "custom-recommended",
  "customV2": "custom-2.0",
  "custom-v2": "custom-2.0",
  "custom-2.0": "custom-2.0",
}


class ShadowReplayError(RuntimeError):
  pass


class MissingModelV2Error(ShadowReplayError):
  pass


@dataclass(frozen=True)
class ShadowReplayOptions:
  stack: str = "custom-2.0"
  preserve_driver_pedals: bool = False
  fallback_v_cruise_kph: float | None = None
  fail_on_unset_cruise: bool = False
  initial_a: float = 0.0


@dataclass(frozen=True)
class ShadowRouteProfile:
  route: str
  route_id: str
  segment: int | None
  samples: int
  duration_s: float
  moving_samples: int
  inferred_cruise_samples: int
  stack_counts: dict[str, int]


@dataclass(frozen=True)
class ShadowReplaySummary:
  route_count: int
  sample_count: int
  moving_sample_count: int
  inferred_cruise_sample_count: int
  correlation: float | None
  mean_abs_error: float
  p90_abs_error: float
  p95_abs_error: float
  opposite_count: int
  opposite_ratio: float
  strong_opposite_count: int
  strong_opposite_ratio: float
  should_stop_moving_count: int
  fcw_count: int
  high_plan_jerk_count: int
  planner_source_counts: dict[str, int]
  sp_source_counts: dict[str, int]
  sp_stack_counts: dict[str, int]
  route_profiles: list[ShadowRouteProfile]
  episodes: list[PlannerTargetEpisode]


@dataclass(frozen=True)
class ShadowPlannerTargetSample(PlannerTargetSample):
  inferred_cruise: bool = False


class ShadowSubMaster(dict):
  def __init__(self, payloads: dict[str, Any], log_mono_time: dict[str, int]):
    super().__init__(payloads)
    self.logMonoTime = dict(log_mono_time)
    self.updated = {service: True for service in payloads}
    self.valid = {service: True for service in payloads}
    self.alive = {service: True for service in payloads}
    self.freq_ok = {service: True for service in payloads}
    self.recv_frame = {service: 0 for service in payloads}
    # Planner updates are triggered by the current modelV2 frame, so every cached input
    # here is fresh enough to replay. A zero timestamp makes the custom adapter treat the
    # model as stale and silently bypass its stop/slowdown policy.
    now = time.monotonic()
    self.recv_time = {service: now for service in payloads}
    self.frame = 0

  def all_checks(self, service_list: list[str] | tuple[str, ...] | None = None) -> bool:
    return True

  def all_alive(self, service_list: list[str] | tuple[str, ...] | None = None) -> bool:
    return True

  def all_freq_ok(self, service_list: list[str] | tuple[str, ...] | None = None) -> bool:
    return True


PlannerFactory = Callable[[Any, Any, float, float], Any]


def main() -> None:
  parser = argparse.ArgumentParser(description="Run longitudinal planner in as-if-engaged shadow mode over manual route logs.")
  parser.add_argument("routes", nargs="+", help="Routes, rlog files, or URLs accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Use qlogs. Mostly useful to verify the modelV2/rlog guard.")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--output", help="Write JSON summary to this path")
  parser.add_argument("--stack", default="customV2", choices=sorted(STACK_ALIASES), help="Stack to shadow")
  parser.add_argument("--preserve-driver-pedals", action="store_true", help="Keep logged gas/brake in planner inputs")
  parser.add_argument("--fallback-v-cruise-kph", type=float,
                      help="Fixed virtual cruise speed when logs have unset cruise; default infers max(vEgo, 40 kph)")
  parser.add_argument("--fail-on-unset-cruise", action="store_true", help="Fail instead of inferring cruise when vCruise is unset")
  parser.add_argument("--large-error", type=float, default=DEFAULT_LARGE_ERROR_THRESHOLD)
  parser.add_argument("--episode-gap", type=float, default=DEFAULT_EPISODE_GAP_S)
  parser.add_argument("--episode-context", type=float, default=DEFAULT_EPISODE_CONTEXT_S)
  parser.add_argument("--high-jerk", type=float, default=DEFAULT_HIGH_JERK_THRESHOLD)
  parser.add_argument("--episodes", type=int, default=12, help="Number of suspicious episodes to show in text output")
  args = parser.parse_args()

  options = ShadowReplayOptions(
    stack=normalize_stack(args.stack),
    preserve_driver_pedals=args.preserve_driver_pedals,
    fallback_v_cruise_kph=args.fallback_v_cruise_kph,
    fail_on_unset_cruise=args.fail_on_unset_cruise,
  )
  read_mode = ReadMode.QLOG if args.qlog else ReadMode.RLOG

  try:
    with OpenpilotPrefix(prefix="drive-lab-shadow-longitudinal"):
      configure_shadow_params(options.stack)
      samples_by_route = {
        route: extract_shadow_samples(route, read_mode, options)
        for route in args.routes
      }
  except ShadowReplayError as exc:
    raise SystemExit(str(exc)) from exc

  summary = summarize_shadow_agreement(
    samples_by_route,
    large_error_threshold=args.large_error,
    episode_gap_s=args.episode_gap,
    episode_context_s=args.episode_context,
    high_jerk_threshold=args.high_jerk,
  )
  payload = asdict(summary)
  if args.output:
    with open(args.output, "w") as f:
      json.dump(payload, f, indent=2)
      f.write("\n")
  print(json.dumps(payload, indent=2) if args.json else render_shadow_summary(summary, max_episodes=args.episodes))


def extract_shadow_samples(route: str, read_mode: ReadMode, options: ShadowReplayOptions,
                           planner_factory: PlannerFactory | None = None) -> list[PlannerTargetSample]:
  route_id, segment = route_identity(route)
  latest_raw: dict[str, Any] = {}
  latest_shadow: dict[str, Any] = {}
  latest_mono_time: dict[str, int] = {}
  car_params = None
  car_params_sp = None
  planner = None
  samples: list[PlannerTargetSample] = []
  latest_car_state_inferred_cruise = False
  seen_model_v2 = False
  route_messages = iter_route_messages(route, read_mode, log_reader_factory=LogReader)

  try:
    for route_msg in route_messages:
      typ = route_msg.typ
      payload = route_msg.payload
      mono_time = route_msg.log_mono_time

      if typ == "modelV2":
        seen_model_v2 = True

      if typ == "carParams":
        car_params = payload
        continue
      if typ == "carParamsSP":
        car_params_sp = payload
        continue
      if typ in REQUIRED_SHADOW_SERVICES or typ in OPTIONAL_SHADOW_SERVICES:
        latest_raw[typ] = payload
        shadow_payload, inferred = shape_shadow_payload(typ, payload, options)
        latest_shadow[typ] = shadow_payload
        latest_mono_time[typ] = mono_time
        if typ == "carState":
          latest_car_state_inferred_cruise = inferred

      if typ != "modelV2" or not shadow_inputs_ready(latest_shadow):
        continue
      if car_params is None or car_params_sp is None:
        continue
      add_optional_defaults(latest_shadow, latest_mono_time, mono_time)
      if planner is None:
        factory = planner_factory or default_planner_factory
        planner = factory(car_params, car_params_sp, float(safe_get(latest_shadow["carState"], "vEgo", 0.0)), options.initial_a)
        configure_shadow_stack(planner, options.stack, car_params, car_params_sp)

      sm = ShadowSubMaster(latest_shadow, latest_mono_time)
      if hasattr(getattr(planner, "sla", None), "update_car_state"):
        planner.sla.update_car_state(sm["carState"])
      planner.update(sm)
      samples.append(build_shadow_sample(
        route, route_id, segment, route_msg.t, latest_raw, latest_shadow, planner,
        latest_car_state_inferred_cruise,
      ))
  except ShadowReplayError:
    if seen_model_v2 or any(route_msg.typ == "modelV2" for route_msg in route_messages):
      raise
    raise MissingModelV2Error(f"{route}: no modelV2 messages found; shadow replay requires rlogs")

  if not seen_model_v2:
    raise MissingModelV2Error(f"{route}: no modelV2 messages found; shadow replay requires rlogs")

  return samples


def summarize_shadow_agreement(samples_by_route: dict[str, list[PlannerTargetSample]],
                               large_error_threshold: float = DEFAULT_LARGE_ERROR_THRESHOLD,
                               episode_gap_s: float = DEFAULT_EPISODE_GAP_S,
                               episode_context_s: float = DEFAULT_EPISODE_CONTEXT_S,
                               high_jerk_threshold: float = DEFAULT_HIGH_JERK_THRESHOLD) -> ShadowReplaySummary:
  samples = [sample for route_samples in samples_by_route.values() for sample in route_samples]
  moving_samples = [sample for sample in samples if sample.v_ego > MOVING_SPEED]
  abs_errors = [abs(sample.plan_a_target - sample.a_ego) for sample in moving_samples]
  opposite_samples = [sample for sample in moving_samples if is_opposite_intent(sample)]
  strong_samples = [sample for sample in moving_samples if is_strong_opposite_intent(sample)]
  high_jerks = high_plan_jerk_pairs(moving_samples, high_jerk_threshold)
  episodes = build_suspicious_episodes(
    moving_samples,
    large_error_threshold=large_error_threshold,
    episode_gap_s=episode_gap_s,
    context_s=episode_context_s,
    high_jerk_threshold=high_jerk_threshold,
  )
  return ShadowReplaySummary(
    route_count=len(samples_by_route),
    sample_count=len(samples),
    moving_sample_count=len(moving_samples),
    inferred_cruise_sample_count=sum(1 for sample in samples if is_inferred_cruise_sample(sample)),
    correlation=_correlation([sample.plan_a_target for sample in moving_samples], [sample.a_ego for sample in moving_samples]),
    mean_abs_error=_mean(abs_errors),
    p90_abs_error=_percentile(abs_errors, 90.0),
    p95_abs_error=_percentile(abs_errors, 95.0),
    opposite_count=len(opposite_samples),
    opposite_ratio=_ratio(len(opposite_samples), len(moving_samples)),
    strong_opposite_count=len(strong_samples),
    strong_opposite_ratio=_ratio(len(strong_samples), len(moving_samples)),
    should_stop_moving_count=sum(1 for sample in moving_samples if sample.plan_should_stop),
    fcw_count=sum(1 for sample in moving_samples if sample.plan_fcw),
    high_plan_jerk_count=len(high_jerks),
    planner_source_counts=dict(Counter(sample.plan_source for sample in moving_samples)),
    sp_source_counts=dict(Counter(sample.sp_source for sample in moving_samples)),
    sp_stack_counts=dict(Counter(sample.sp_stack for sample in moving_samples)),
    route_profiles=[build_shadow_route_profile(route, route_samples) for route, route_samples in samples_by_route.items()],
    episodes=episodes,
  )


def render_shadow_summary(summary: ShadowReplaySummary, max_episodes: int = 12) -> str:
  lines = [
    "Drive Lab shadow longitudinal agreement",
    f"routes={summary.route_count} samples={summary.sample_count} moving={summary.moving_sample_count} "
    + f"inferred_cruise={summary.inferred_cruise_sample_count}",
    f"shadow aTarget vs aEgo: corr={_format_optional(summary.correlation)} mean_abs={summary.mean_abs_error:.3f} "
    + f"p90_abs={summary.p90_abs_error:.3f} p95_abs={summary.p95_abs_error:.3f} m/s^2",
    f"opposite_intent={summary.opposite_count} ({summary.opposite_ratio:.3%}) "
    + f"strong={summary.strong_opposite_count} ({summary.strong_opposite_ratio:.3%})",
    f"shouldStop_moving={summary.should_stop_moving_count} fcw={summary.fcw_count} "
    + f"high_plan_jerk={summary.high_plan_jerk_count}",
    "Planner sources: " + _format_counts(summary.planner_source_counts),
    "SP sources: " + _format_counts(summary.sp_source_counts),
    "SP stacks: " + _format_counts(summary.sp_stack_counts),
    "Routes:",
  ]
  for profile in summary.route_profiles:
    segment = f"--{profile.segment}" if profile.segment is not None else ""
    lines.append(
      f"  {profile.route_id}{segment}: samples {profile.samples}, moving {profile.moving_samples}, "
      + f"inferred cruise {profile.inferred_cruise_samples}, duration {profile.duration_s:.1f}s, "
      + "stacks " + _format_counts(profile.stack_counts)
    )
  lines.append("Suspicious episodes:")
  if not summary.episodes:
    lines.append("  none")
  for episode in summary.episodes[:max_episodes]:
    segment = f"--{episode.segment}" if episode.segment is not None else ""
    lines.append(
      f"  {episode.route_id}{segment} {episode.start_time_s:.1f}-{episode.end_time_s:.1f}s "
      + f"n={episode.sample_count} opp={episode.opposite_count} strong={episode.strong_opposite_count} "
      + f"max_err={episode.max_abs_error:.2f} sources={_format_counts(episode.planner_sources)} "
      + f"gas={episode.driver_gas_count} brake={episode.driver_brake_count} lead={episode.lead_ratio:.2f} "
      + f"min_d={_format_optional(episode.min_lead_d_rel)} min_vrel={_format_optional(episode.min_lead_v_rel)} "
      + f"lead_flips={episode.lead_status_flips} source_flips={episode.plan_source_flips} "
      + f"plan_span={episode.plan_span:.2f} high_jerk={episode.high_plan_jerk_count}"
    )
  return "\n".join(lines)


def build_shadow_route_profile(route: str, samples: list[PlannerTargetSample]) -> ShadowRouteProfile:
  route_id, segment = route_identity(route)
  return ShadowRouteProfile(
    route=route,
    route_id=route_id,
    segment=segment,
    samples=len(samples),
    duration_s=route_duration(samples),
    moving_samples=sum(1 for sample in samples if sample.v_ego > MOVING_SPEED),
    inferred_cruise_samples=sum(1 for sample in samples if is_inferred_cruise_sample(sample)),
    stack_counts=dict(Counter(sample.sp_stack for sample in samples)),
  )


def build_shadow_sample(route: str, route_id: str, segment: int | None, t: float, latest_raw: dict[str, Any],
                        latest_shadow: dict[str, Any], planner: Any, inferred_cruise: bool) -> PlannerTargetSample:
  car_state_raw = latest_raw["carState"]
  car_state_shadow = latest_shadow["carState"]
  radar_state = latest_shadow["radarState"]
  lead = safe_get(radar_state, "leadOne")
  source = format_enum(getattr(planner, "longitudinal_plan_source", getattr(getattr(planner, "mpc", None), "source", "unknown")))
  sp_source = format_enum(getattr(planner, "source", source))
  stack = stack_display_name(getattr(planner, "longitudinal_stack_actuated_stack", "unknown"))
  return ShadowPlannerTargetSample(
    route=route,
    route_id=route_id,
    segment=segment,
    t=t,
    v_ego=float(safe_get(car_state_raw, "vEgo", 0.0)),
    a_ego=float(safe_get(car_state_raw, "aEgo", 0.0)),
    gas_pressed=bool(safe_get(car_state_raw, "gasPressed", False)),
    brake_pressed=bool(safe_get(car_state_raw, "brakePressed", False)),
    standstill=bool(safe_get(car_state_raw, "standstill", False)),
    selfdrive_enabled=True,
    selfdrive_active=True,
    long_active=True,
    long_control_state="pid",
    v_cruise_kph=_finite_or_none(safe_get(car_state_shadow, "vCruise")),
    plan_a_target=float(getattr(planner, "output_a_target", 0.0)),
    plan_source=source,
    plan_should_stop=bool(getattr(planner, "output_should_stop", False)),
    plan_fcw=bool(getattr(planner, "fcw", False)),
    sp_a_target=float(getattr(planner, "output_a_target", 0.0)),
    sp_source=sp_source,
    sp_stack=stack,
    lead_status=bool(safe_get(lead, "present", False)),
    lead_d_rel=_finite_or_none(safe_get(lead, "dRel")),
    lead_v_rel=_finite_or_none(safe_get(lead, "vRel")),
    model_desired_accel=_finite_or_none(safe_get(latest_shadow["modelV2"], "action.desiredAcceleration")),
    model_should_stop=bool(safe_get(latest_shadow["modelV2"], "action.shouldStop", False)),
    inferred_cruise=bool(inferred_cruise),
  )


def is_inferred_cruise_sample(sample: PlannerTargetSample) -> bool:
  return bool(getattr(sample, "inferred_cruise", False))


def shape_shadow_payload(typ: str, payload: Any, options: ShadowReplayOptions) -> tuple[Any, bool]:
  shadow = payload_copy(payload)
  inferred_cruise = False
  if typ == "selfdriveState":
    set_attr(shadow, "enabled", True)
    set_attr(shadow, "active", True)
  elif typ == "carControl":
    set_attr(shadow, "enabled", True)
    set_attr(shadow, "longActive", True)
    cruise_control = safe_get(shadow, "cruiseControl")
    if cruise_control is not None:
      set_attr(cruise_control, "override", False)
  elif typ == "controlsState":
    set_attr(shadow, "longControlState", "pid")
    set_attr(shadow, "forceDecel", False)
  elif typ == "carState":
    inferred_cruise = apply_virtual_cruise(shadow, options)
    if not options.preserve_driver_pedals:
      set_attr(shadow, "gasPressed", False)
      set_attr(shadow, "brakePressed", False)
  return shadow, inferred_cruise


def apply_virtual_cruise(car_state: Any, options: ShadowReplayOptions) -> bool:
  v_cruise = _finite_or_none(safe_get(car_state, "vCruise"))
  if v_cruise is not None and 0.0 < v_cruise < UNSET_CRUISE_KPH:
    return False
  if options.fail_on_unset_cruise:
    raise ShadowReplayError("encountered unset vCruise; pass --fallback-v-cruise-kph or omit --fail-on-unset-cruise")
  inferred = options.fallback_v_cruise_kph
  if inferred is None:
    inferred = max(DEFAULT_MIN_INFERRED_CRUISE_KPH, float(safe_get(car_state, "vEgo", 0.0)) * CV.MS_TO_KPH)
  set_attr(car_state, "vCruise", float(inferred))
  set_attr(car_state, "vCruiseCluster", float(inferred))
  return True


def shadow_inputs_ready(latest_shadow: dict[str, Any]) -> bool:
  return all(service in latest_shadow for service in REQUIRED_SHADOW_SERVICES)


def add_optional_defaults(latest_shadow: dict[str, Any], latest_mono_time: dict[str, int], mono_time: int) -> None:
  for service in OPTIONAL_SHADOW_SERVICES:
    if service not in latest_shadow:
      latest_shadow[service] = default_payload(service)
      latest_mono_time[service] = mono_time


def default_payload(service: str) -> Any:
  msg = messaging.new_message(service)
  return getattr(msg, service)


def default_planner_factory(CP: Any, CP_SP: Any, init_v: float, init_a: float) -> Any:
  try:
    from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
  except ModuleNotFoundError as exc:
    raise ShadowReplayError(
      "LongitudinalPlanner import failed; build generated longitudinal MPC modules before shadow replay: " + str(exc)
    ) from exc
  return LongitudinalPlanner(CP, CP_SP, init_v=init_v, init_a=init_a)


def configure_shadow_stack(planner: Any, stack: str, CP: Any, CP_SP: Any) -> None:
  # The restart retired stack multiplexing. Preserve the legacy fallback below for old
  # checkouts, but make current planners report the stack selected by this replay.
  planner.longitudinal_stack_actuated_stack = stack
  if not hasattr(planner, "longitudinal_stack_resolution"):
    return
  from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import resolve_longitudinal_stack

  planner.longitudinal_stack_resolution = resolve_longitudinal_stack(stack, CP, CP_SP)
  if hasattr(planner, "_make_custom_longitudinal_stack"):
    planner.custom_longitudinal_stack = planner._make_custom_longitudinal_stack(planner.longitudinal_stack_resolution.resolved_stack)


def configure_shadow_params(stack: str) -> None:
  """Materialize fresh-install defaults inside the isolated replay prefix."""
  params = Params()
  for key in params.all_keys():
    if params.get(key) is None and (default := params.get_default_value(key)) is not None:
      params.put(key, default, block=True)
  params.put_bool("CustomLongitudinalEnabled", stack != "sunnypilot-current", block=True)


def payload_copy(payload: Any) -> Any:
  as_builder = getattr(payload, "as_builder", None)
  if callable(as_builder):
    return as_builder()
  as_reader = getattr(payload, "as_reader", None)
  if callable(as_reader):
    reader_as_builder = getattr(as_reader(), "as_builder", None)
    if callable(reader_as_builder):
      return reader_as_builder()
  return copy.deepcopy(payload)


def set_attr(obj: Any, name: str, value: Any) -> None:
  try:
    setattr(obj, name, value)
  except Exception:
    pass


def normalize_stack(stack: str) -> str:
  return STACK_ALIASES.get(str(stack), str(stack))


def stack_display_name(stack: object) -> str:
  if isinstance(stack, bytes):
    text = stack.decode(errors="ignore")
  elif isinstance(stack, str):
    text = stack
  else:
    text = format_enum(stack)
  return {
    "sunnypilot-current": "sunnypilotCurrent",
    "custom-recommended": "customRecommended",
    "custom-2.0": "customV2",
  }.get(text, text)


if __name__ == "__main__":
  main()
