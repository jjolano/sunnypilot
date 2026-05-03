from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class TimelineEvent:
  time_s: float
  category: str
  detail: str


@dataclass(frozen=True)
class SignalStats:
  name: str
  minimum: float | None = None
  maximum: float | None = None
  final: float | None = None


@dataclass(frozen=True)
class EventAttribution:
  likely_cause: str
  evidence: list[str]


@dataclass(frozen=True)
class EventWindowSummary:
  event_time_s: float
  start_time_s: float
  end_time_s: float
  timeline: list[TimelineEvent]
  attribution: EventAttribution
  stats: list[SignalStats]


def safe_get(obj: Any, path: str, default: Any = None) -> Any:
  cur = obj
  for part in path.split('.'):
    try:
      cur = getattr(cur, part)
    except (AttributeError, TypeError):
      return default
  return cur


def format_enum(value: Any) -> str:
  if value is None:
    return "unknown"
  if isinstance(value, IntEnum):
    return value.name
  if hasattr(value, "name"):
    return str(value.name)
  return str(value).split('.')[-1]


def msg_type(msg: Any) -> str:
  which = getattr(msg, "which", None)
  if callable(which):
    return which()
  return str(which) if which is not None else "unknown"


def msg_payload(msg: Any) -> Any:
  typ = msg_type(msg)
  return safe_get(msg, typ, msg)


def msg_time_s(msg: Any, base_mono_time: int | None = None) -> float:
  log_mono_time = int(getattr(msg, "logMonoTime", 0))
  if base_mono_time is None:
    base_mono_time = 0
  return (log_mono_time - base_mono_time) / 1e9


def find_bookmark_times(msgs: list[Any], base_mono_time: int | None = None) -> list[float]:
  return [msg_time_s(m, base_mono_time) for m in msgs if msg_type(m) == "userBookmark"]


def select_event_time(msgs: list[Any], requested_time_s: float | None = None, nearest_bookmark: bool = False) -> float:
  if not msgs:
    raise ValueError("no log messages supplied")

  ordered_msgs = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  base_mono_time = int(getattr(ordered_msgs[0], "logMonoTime", 0))
  if nearest_bookmark:
    bookmark_times = find_bookmark_times(ordered_msgs, base_mono_time)
    if not bookmark_times:
      raise ValueError("no userBookmark messages found in supplied logs")
    if requested_time_s is None:
      return bookmark_times[-1]
    return min(bookmark_times, key=lambda t: abs(t - requested_time_s))

  if requested_time_s is None:
    raise ValueError("provide --time or --nearest-bookmark")
  return requested_time_s


def summarize_window(msgs: list[Any], event_time_s: float, before_s: float, after_s: float) -> EventWindowSummary:
  if not msgs:
    raise ValueError("no log messages supplied")

  msgs = sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  base_mono_time = int(getattr(msgs[0], "logMonoTime", 0))
  start_s = event_time_s - before_s
  end_s = event_time_s + after_s
  window_msgs = [m for m in msgs if start_s <= msg_time_s(m, base_mono_time) <= end_s]

  timeline: list[TimelineEvent] = []
  samples: dict[str, list[float]] = {
    "vEgo": [],
    "aTarget": [],
    "lead.dRel": [],
    "lead.vRel": [],
  }
  last_values: dict[str, Any] = {}
  attribution_facts: dict[str, Any] = {
    "event_time_s": event_time_s,
    "planner_sources": [],
    "sp_sources": [],
    "sp_samples": [],
    "lead_present": False,
    "lead_braking": False,
    "driver_samples": [],
    "lead_times": [],
    "braking_times": [],
    "lead_gaps": [],
    "a_targets": [],
    "model_action_should_stop": False,
    "plan_model_should_stop": False,
    "speed_limit_active": False,
    "scc_map_active": False,
    "scc_vision_active": False,
  }
  lead_active = False

  def add_change(t: float, key: str, value: Any, category: str, detail: str) -> None:
    previous = last_values.get(key, object())
    if value != previous:
      timeline.append(TimelineEvent(t, category, detail))
      last_values[key] = value

  for msg in window_msgs:
    typ = msg_type(msg)
    payload = msg_payload(msg)
    t = msg_time_s(msg, base_mono_time)

    if typ == "userBookmark":
      timeline.append(TimelineEvent(t, "marker", "user bookmark"))
    elif typ == "carState":
      v_ego = safe_get(payload, "vEgo")
      if _finite_number(v_ego):
        samples["vEgo"].append(float(v_ego))
      cruise = safe_get(payload, "vCruise")
      if _finite_number(cruise):
        add_change(t, "carState.vCruise", round(float(cruise), 1), "car", f"cruise setpoint {float(cruise):.1f} kph")
      for field, label in (("brakePressed", "brake pressed"), ("gasPressed", "gas pressed"), ("standstill", "standstill")):
        value = bool(safe_get(payload, field, False))
        add_change(t, f"carState.{field}", value, "car", f"{label}: {value}")
      attribution_facts["driver_samples"].append({
        "time_s": t,
        "brake": bool(safe_get(payload, "brakePressed", False)),
        "gas": bool(safe_get(payload, "gasPressed", False)),
      })
    elif typ == "selfdriveState":
      for field in ("enabled", "active", "experimentalMode", "personality"):
        value = safe_get(payload, field)
        if value is not None:
          add_change(t, f"selfdriveState.{field}", value, "selfdrive", f"{field}: {format_enum(value)}")
    elif typ == "radarState":
      lead = safe_get(payload, "leadOne")
      lead_status = bool(safe_get(lead, "status", False))
      lead_active = lead_status
      add_change(t, "radarState.leadOne.status", lead_status, "lead", f"leadOne status: {lead_status}")
      d_rel = safe_get(lead, "dRel")
      v_rel = safe_get(lead, "vRel")
      if lead_status:
        attribution_facts["lead_present"] = True
        attribution_facts["lead_times"].append(t)
      if lead_status and _finite_number(d_rel):
        attribution_facts["lead_gaps"].append(float(d_rel))
      if lead_status and _finite_number(d_rel):
        samples["lead.dRel"].append(float(d_rel))
      if lead_status and _finite_number(v_rel):
        samples["lead.vRel"].append(float(v_rel))
    elif typ == "longitudinalPlan":
      source = format_enum(safe_get(payload, "longitudinalPlanSource"))
      add_change(t, "longitudinalPlan.source", source, "planner", f"plan source: {source}")
      should_stop = bool(safe_get(payload, "shouldStop", False))
      add_change(t, "longitudinalPlan.shouldStop", should_stop, "planner", f"shouldStop: {should_stop}")
      if should_stop and _is_model_stop_source(source):
        attribution_facts["plan_model_should_stop"] = True
      fcw = bool(safe_get(payload, "fcw", False))
      add_change(t, "longitudinalPlan.fcw", fcw, "planner", f"fcw: {fcw}")
      a_target = safe_get(payload, "aTarget")
      if source != "unknown":
        attribution_facts["planner_sources"].append(source)
      if _finite_number(a_target):
        a_target_float = float(a_target)
        attribution_facts["a_targets"].append(a_target_float)
        if lead_active and _is_braking([a_target_float]):
          attribution_facts["lead_braking"] = True
        if _is_braking([a_target_float]):
          attribution_facts["braking_times"].append(t)
      if _finite_number(a_target):
        samples["aTarget"].append(float(a_target))
    elif typ == "longitudinalPlanSP":
      source = format_enum(safe_get(payload, "longitudinalPlanSource"))
      add_change(t, "longitudinalPlanSP.source", source, "sunnypilot", f"SP source: {source}")
      if source != "unknown":
        attribution_facts["sp_sources"].append(source)
      sp_sample = {
        "time_s": t,
        "source": source,
        "speed_limit_active": False,
        "scc_map_active": False,
        "scc_vision_active": False,
      }
      for path, label, fact_key in (
        ("smartCruiseControl.vision.active", "SCC vision active", "scc_vision_active"),
        ("smartCruiseControl.map.active", "SCC map active", "scc_map_active"),
        ("speedLimit.assist.active", "speed-limit assist active", "speed_limit_active"),
        ("speedLimit.assist.autoCruiseEnabled", "speed-limit auto-cruise enabled", None),
      ):
        value = safe_get(payload, path)
        if value is not None:
          active = bool(value)
          add_change(t, f"longitudinalPlanSP.{path}", active, "sunnypilot", f"{label}: {active}")
          if fact_key is not None and active:
            attribution_facts[fact_key] = True
          if fact_key is not None:
            sp_sample[fact_key] = active
      attribution_facts["sp_samples"].append(sp_sample)
    elif typ == "modelV2":
      desired_accel = safe_get(payload, "action.desiredAcceleration")
      if _finite_number(desired_accel):
        add_change(t, "modelV2.action.desiredAcceleration", round(float(desired_accel), 1), "model", f"desired accel {float(desired_accel):.1f} m/s^2")
      should_stop = safe_get(payload, "action.shouldStop")
      if should_stop is not None:
        add_change(t, "modelV2.action.shouldStop", bool(should_stop), "model", f"model shouldStop: {bool(should_stop)}")
      if bool(should_stop):
        attribution_facts["model_action_should_stop"] = True
    elif typ == "onroadEvents":
      events = safe_get(payload, "events")
      if events is None and not isinstance(payload, str | bytes):
        try:
          iter(payload)
          events = payload
        except TypeError:
          events = []
      names = _event_names(events or [])
      add_change(t, "onroadEvents.events", tuple(names), "event", "events: " + ", ".join(names) if names else "events cleared")

  return EventWindowSummary(event_time_s, start_s, end_s, timeline, _build_attribution(attribution_facts), _build_stats(samples))


def render_summary(summary: EventWindowSummary) -> str:
  lines = [
    f"Drive Lab event at {summary.event_time_s:.2f}s",
    f"Window: {summary.start_time_s:.2f}s to {summary.end_time_s:.2f}s",
    "",
    "Timeline:",
  ]
  if summary.timeline:
    for item in sorted(summary.timeline, key=lambda e: e.time_s):
      lines.append(f"  {item.time_s:8.2f}s  {item.category:11s} {item.detail}")
  else:
    lines.append("  no notable state changes found")

  lines.append("")
  lines.append("Attribution:")
  lines.append(f"  likely cause: {summary.attribution.likely_cause}")
  for item in summary.attribution.evidence:
    lines.append(f"  evidence: {item}")

  lines.append("")
  lines.append("Signal stats:")
  for stat in summary.stats:
    if stat.minimum is None or stat.maximum is None or stat.final is None:
      continue
    lines.append(f"  {stat.name:12s} min {stat.minimum:8.3f}  max {stat.maximum:8.3f}  final {stat.final:8.3f}")
  return "\n".join(lines)


def _build_attribution(facts: dict[str, Any]) -> EventAttribution:
  evidence = _attribution_evidence(facts)
  driver_sample = _event_local_sample(facts["driver_samples"], facts["event_time_s"])
  sp_sample = _event_local_sp_sample(facts["sp_samples"], facts["event_time_s"])
  sp_source_cause = _sp_source_cause(sp_sample)
  if driver_sample is not None and (driver_sample["brake"] or driver_sample["gas"]):
    cause = "driver"
  elif _has_lead_source(facts["planner_sources"]) or facts["lead_braking"] or _has_correlated_lead_braking(facts):
    cause = "lead"
  elif facts["model_action_should_stop"] or facts["plan_model_should_stop"]:
    cause = "model_stop"
  elif sp_source_cause is not None:
    cause = sp_source_cause
  elif sp_sample is not None and sp_sample["speed_limit_active"]:
    cause = "speed_limit"
  elif sp_sample is not None and sp_sample["scc_map_active"]:
    cause = "scc_map"
  elif sp_sample is not None and sp_sample["scc_vision_active"]:
    cause = "scc_vision"
  elif facts["planner_sources"] or facts["sp_sources"]:
    cause = "planner_source"
  else:
    cause = "unknown"
    if not evidence:
      evidence = ["no longitudinal attribution signals found"]
  return EventAttribution(cause, evidence)


def _attribution_evidence(facts: dict[str, Any]) -> list[str]:
  evidence: list[str] = []
  driver_sample = _event_local_sample(facts["driver_samples"], facts["event_time_s"])
  if driver_sample is not None and driver_sample["brake"]:
    evidence.append("driver brake pressed")
  if driver_sample is not None and driver_sample["gas"]:
    evidence.append("driver gas pressed")
  for source in _unique_ordered(facts["planner_sources"]):
    evidence.append(f"planner source {source}")
  for source in _unique_ordered(facts["sp_sources"]):
    evidence.append(f"SP source {source}")
  if facts["lead_gaps"]:
    evidence.append(f"lead gap min {min(facts['lead_gaps']):.3f} m")
  if facts["model_action_should_stop"]:
    evidence.append("model action shouldStop true")
  if facts["plan_model_should_stop"]:
    evidence.append("plan shouldStop true")
  if facts["speed_limit_active"]:
    evidence.append("speed-limit assist active")
  if facts["scc_map_active"]:
    evidence.append("SCC map active")
  if facts["scc_vision_active"]:
    evidence.append("SCC vision active")
  if facts["a_targets"]:
    evidence.append(f"aTarget min {min(facts['a_targets']):.3f} m/s^2")
  return evidence


def _has_lead_source(sources: list[str]) -> bool:
  return any(source.lower().startswith("lead") for source in sources)


def _is_model_stop_source(source: str) -> bool:
  return source.lower() in ("model", "e2e")


def _sp_source_cause(sample: dict[str, Any] | None) -> str | None:
  if sample is None:
    return None
  normalized = str(sample["source"]).lower().replace("_", "").replace("-", "")
  if normalized in ("speedlimit", "speedlimitassist"):
    return "speed_limit"
  if normalized in ("sccmap", "map"):
    return "scc_map"
  if normalized in ("sccvision", "vision"):
    return "scc_vision"
  return None


def _event_local_sp_sample(samples: list[dict[str, Any]], event_time_s: float) -> dict[str, Any] | None:
  return _event_local_sample(samples, event_time_s)


def _event_local_sample(samples: list[dict[str, Any]], event_time_s: float) -> dict[str, Any] | None:
  if not samples:
    return None
  prior_samples = [sample for sample in samples if float(sample["time_s"]) <= event_time_s]
  if prior_samples:
    return max(prior_samples, key=lambda sample: float(sample["time_s"]))
  return min(samples, key=lambda sample: float(sample["time_s"]))


def _is_braking(a_targets: list[float]) -> bool:
  return bool(a_targets) and min(a_targets) < -0.05


def _has_correlated_lead_braking(facts: dict[str, Any]) -> bool:
  lead_times = facts["lead_times"]
  braking_times = facts["braking_times"]
  return any(lead_t == braking_t for lead_t in lead_times for braking_t in braking_times)


def _unique_ordered(values: list[str]) -> list[str]:
  seen = set()
  unique = []
  for value in values:
    if value not in seen:
      seen.add(value)
      unique.append(value)
  return unique


def _build_stats(samples: dict[str, list[float]]) -> list[SignalStats]:
  stats = []
  for name, values in samples.items():
    if values:
      stats.append(SignalStats(name, min(values), max(values), values[-1]))
  return stats


def _event_names(events: Any) -> list[str]:
  names = []
  for event in events:
    name = safe_get(event, "name")
    names.append(format_enum(name))
  return names


def _finite_number(value: Any) -> bool:
  return isinstance(value, int | float) and isfinite(float(value))
