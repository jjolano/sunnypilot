from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from openpilot.tools.drive_lab.route_analysis import build_route_messages
from openpilot.tools.drive_lab.timeline import safe_get


@dataclass(frozen=True)
class ProfileRange:
  low: float
  high: float


@dataclass(frozen=True)
class LongitudinalProfile:
  source: str
  sample_count: int
  ego_speed: ProfileRange
  cruise_speed: ProfileRange
  lead_gap: ProfileRange
  closing_speed: ProfileRange
  lead_decel: ProfileRange
  stopped_lead_gap: ProfileRange
  lead_pullaway_speed: ProfileRange

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LongitudinalProfile:
    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
      ego_speed=_profile_range_from_dict(data["ego_speed"] if "ego_speed" in data else data["egoSpeed"]),
      cruise_speed=_profile_range_from_dict(data["cruise_speed"] if "cruise_speed" in data else data["cruiseSpeed"]),
      lead_gap=_profile_range_from_dict(data["lead_gap"] if "lead_gap" in data else data["leadGap"]),
      closing_speed=_profile_range_from_dict(data["closing_speed"] if "closing_speed" in data else data["closingSpeed"]),
      lead_decel=_profile_range_from_dict(data["lead_decel"] if "lead_decel" in data else data["leadDecel"]),
      stopped_lead_gap=_profile_range_from_dict(data["stopped_lead_gap"] if "stopped_lead_gap" in data else data["stoppedLeadGap"]),
      lead_pullaway_speed=_profile_range_from_dict(data["lead_pullaway_speed"] if "lead_pullaway_speed" in data else data["leadPullawaySpeed"]),
    )


def build_longitudinal_profile(msgs: list[Any], source: str = "unknown", already_sorted: bool = False) -> LongitudinalProfile:
  msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  route_msgs = build_route_messages(msgs)
  car_state_by_time: list[tuple[float, float, bool]] = []
  ego_speeds: list[float] = []
  cruise_speeds: list[float] = []
  lead_gaps: list[float] = []
  closing_speeds: list[float] = []
  lead_decels: list[float] = []
  stopped_lead_gaps: list[float] = []
  lead_pullaway_speeds: list[float] = []

  prev_lead_time: float | None = None
  prev_lead_speed: float | None = None

  for route_msg in route_msgs:
    typ = route_msg.typ
    payload = route_msg.payload
    t = route_msg.t
    if typ == "carState":
      v_ego = safe_get(payload, "vEgo")
      if _finite_positive(v_ego, allow_zero=True):
        v_ego_float = float(v_ego)
        ego_speeds.append(v_ego_float)
        car_state_by_time.append((t, v_ego_float, bool(safe_get(payload, "standstill", v_ego_float < 0.3))))
      v_cruise_kph = safe_get(payload, "vCruise")
      if _finite_positive(v_cruise_kph) and float(v_cruise_kph) < 255.0:
        cruise_speeds.append(float(v_cruise_kph) / 3.6)
    elif typ == "radarState":
      lead = safe_get(payload, "leadOne")
      if not bool(safe_get(lead, "status", False)):
        prev_lead_time = None
        prev_lead_speed = None
        continue

      d_rel = safe_get(lead, "dRel")
      v_rel = safe_get(lead, "vRel")
      v_lead = safe_get(lead, "vLead")
      if _finite_positive(d_rel):
        lead_gaps.append(float(d_rel))
      if _finite_number(v_rel):
        closing_speeds.append(max(0.0, -float(v_rel)))
      if _finite_positive(v_lead, allow_zero=True):
        v_lead_float = float(v_lead)
        if prev_lead_time is not None and prev_lead_speed is not None:
          dt = t - prev_lead_time
          if dt > 1e-3:
            lead_accel = (v_lead_float - prev_lead_speed) / dt
            if lead_accel < 0.0:
              lead_decels.append(-lead_accel)
        prev_lead_time = t
        prev_lead_speed = v_lead_float

        ego_speed, standstill = _nearest_car_state(car_state_by_time, t)
        if d_rel is not None and standstill and v_lead_float < 0.5 and _finite_positive(d_rel):
          stopped_lead_gaps.append(float(d_rel))
        if d_rel is not None and ego_speed < 1.0 and 0.5 <= v_lead_float <= 6.0 and _finite_positive(d_rel) and float(d_rel) < 20.0:
          lead_pullaway_speeds.append(v_lead_float)

  return LongitudinalProfile(
    source=source,
    sample_count=len(msgs),
    ego_speed=_percentile_range(ego_speeds, (8.0, 24.0), low_pct=15.0, high_pct=90.0),
    cruise_speed=_percentile_range(cruise_speeds, (5.0, 15.0), low_pct=15.0, high_pct=90.0),
    lead_gap=_percentile_range(lead_gaps, (25.0, 70.0), low_pct=15.0, high_pct=85.0),
    closing_speed=_percentile_range(closing_speeds, (0.0, 4.0), low_pct=50.0, high_pct=95.0),
    lead_decel=_percentile_range(lead_decels, (1.5, 3.5), low_pct=50.0, high_pct=95.0),
    stopped_lead_gap=_percentile_range(stopped_lead_gaps, (4.5, 8.0), low_pct=10.0, high_pct=90.0),
    lead_pullaway_speed=_percentile_range(lead_pullaway_speeds, (1.0, 3.5), low_pct=10.0, high_pct=90.0),
  )


def load_profile(path: str | Path) -> LongitudinalProfile:
  with open(path) as f:
    return LongitudinalProfile.from_dict(json.load(f))


def save_profile(profile: LongitudinalProfile, path: str | Path) -> None:
  with open(path, "w") as f:
    json.dump(profile.to_dict(), f, indent=2)
    f.write("\n")


def render_profile(profile: LongitudinalProfile) -> str:
  lines = [f"Drive Lab longitudinal profile: {profile.source}", f"Samples: {profile.sample_count}", ""]
  for name in ("ego_speed", "cruise_speed", "lead_gap", "closing_speed", "lead_decel", "stopped_lead_gap", "lead_pullaway_speed"):
    value = getattr(profile, name)
    lines.append(f"{name:20s} {value.low:8.3f} to {value.high:8.3f}")
  return "\n".join(lines)


def _nearest_car_state(samples: list[tuple[float, float, bool]], t: float) -> tuple[float, bool]:
  if not samples:
    return 0.0, False

  lo = 0
  hi = len(samples)
  while lo < hi:
    mid = (lo + hi) // 2
    if samples[mid][0] < t:
      lo = mid + 1
    else:
      hi = mid

  if lo == 0:
    best = samples[0]
  elif lo == len(samples):
    best = samples[-1]
  else:
    before = samples[lo - 1]
    after = samples[lo]
    best = before if abs(before[0] - t) <= abs(after[0] - t) else after

  if abs(best[0] - t) > 0.5:
    return 0.0, False
  return best[1], best[2]


def _percentile_range(values: list[float], fallback: tuple[float, float], low_pct: float, high_pct: float) -> ProfileRange:
  clean = [float(v) for v in values if _finite_positive(v, allow_zero=True)]
  if len(clean) < 5:
    return ProfileRange(*fallback)
  low = float(np.percentile(clean, low_pct))
  high = float(np.percentile(clean, high_pct))
  if high - low < 0.5:
    mid = 0.5 * (low + high)
    low = mid - 0.25
    high = mid + 0.25
  return ProfileRange(max(0.0, low), max(0.0, high))


def _profile_range_from_dict(data: dict[str, Any]) -> ProfileRange:
  return ProfileRange(float(data["low"]), float(data["high"]))


def _finite_positive(value: Any, allow_zero: bool = False) -> bool:
  return _finite_number(value) and (float(value) >= 0.0 if allow_zero else float(value) > 0.0)


def _finite_number(value: Any) -> bool:
  return isinstance(value, int | float) and isfinite(float(value))


# ── Lateral profile ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LateralProfile:
  source: str
  sample_count: int
  ego_speed: ProfileRange
  curvature: ProfileRange
  lane_confidence: ProfileRange
  roll: ProfileRange

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> LateralProfile:
    return cls(
      source=str(data.get("source", "unknown")),
      sample_count=int(data.get("sample_count", data.get("sampleCount", 0))),
      ego_speed=_profile_range_from_dict(data["ego_speed"] if "ego_speed" in data else data["egoSpeed"]),
      curvature=_profile_range_from_dict(data["curvature"]),
      lane_confidence=_profile_range_from_dict(data["lane_confidence"] if "lane_confidence" in data else data["laneConfidence"]),
      roll=_profile_range_from_dict(data["roll"]),
    )


def build_lateral_profile(msgs: list[Any], source: str = "unknown", already_sorted: bool = False) -> LateralProfile:
  msgs = list(msgs) if already_sorted else sorted(msgs, key=lambda m: int(getattr(m, "logMonoTime", 0)))
  route_msgs = build_route_messages(msgs)
  ego_speeds: list[float] = []
  curvatures: list[float] = []
  lane_confidences: list[float] = []
  rolls: list[float] = []

  for route_msg in route_msgs:
    typ = route_msg.typ
    payload = route_msg.payload
    if typ == "carState":
      v_ego = safe_get(payload, "vEgo")
      if _finite_positive(v_ego, allow_zero=True):
        ego_speeds.append(float(v_ego))
    elif typ == "controlsState":
      k = safe_get(payload, "desiredCurvature")
      if _finite_number(k):
        curvatures.append(abs(float(k)))
      k_actual = safe_get(payload, "curvature")
      if _finite_number(k_actual):
        curvatures.append(abs(float(k_actual)))
    elif typ == "modelV2":
      probs = safe_get(payload, "laneLineProbs")
      if probs is not None and len(probs) >= 2:
        left = float(probs[0]) if _finite_number(probs[0]) else 0.0
        right = float(probs[1]) if _finite_number(probs[1]) else 0.0
        conf = min(max(left, right), 1.0)
        if conf > 0.0:
          lane_confidences.append(conf)
    elif typ == "liveParameters":
      r = safe_get(payload, "roll")
      if _finite_number(r):
        rolls.append(abs(float(r)))

  return LateralProfile(
    source=source,
    sample_count=len(msgs),
    ego_speed=_percentile_range(ego_speeds, (10.0, 25.0), low_pct=5.0, high_pct=95.0),
    curvature=_percentile_range(curvatures, (0.0005, 0.003), low_pct=5.0, high_pct=95.0),
    lane_confidence=_percentile_range(lane_confidences, (0.5, 1.0), low_pct=5.0, high_pct=95.0),
    roll=_percentile_range(rolls, (0.0, 0.05), low_pct=5.0, high_pct=95.0),
  )


def render_lateral_profile(profile: LateralProfile) -> str:
  lines = [f"Drive Lab lateral profile: {profile.source}", f"Samples: {profile.sample_count}", ""]
  for name in ("ego_speed", "curvature", "lane_confidence", "roll"):
    value = getattr(profile, name)
    lines.append(f"{name:20s} {value.low:8.4f} to {value.high:8.4f}")
  return "\n".join(lines)


def load_lateral_profile(path: str | Path) -> LateralProfile:
  with open(path) as f:
    return LateralProfile.from_dict(json.load(f))


def save_lateral_profile(profile: LateralProfile, path: str | Path) -> None:
  with open(path, "w") as f:
    json.dump(profile.to_dict(), f, indent=2)
    f.write("\n")


