"""Shadow-only lead path clearance prediction.

Phase 1 intentionally does not change actuation. It only computes debug fields
for leads that appear to be exiting the ego path before ego reaches the conflict
point.

Coordinate convention: ``radarState`` publishes planner ``yRel``. Vision-only
leads get that from ``-modelV2.leadsV3[i].y[0]`` in ``radard.py``, so model lead
lateral trajectories are normalized here with the same sign flip before they are
compared with path-relative lead state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_APPLY = "apply"
VALID_MODES = {MODE_OFF, MODE_SHADOW, MODE_APPLY}

PATH_CLEAR_Y = 1.7
PATH_CLEAR_SUSTAIN_TOL = 0.2
MIN_LATERAL_DELTA = 0.35
MIN_CLEAR_MARGIN_S = 0.7
MIN_STATE_CONFIDENCE = 0.60
MIN_MODEL_PROB = 0.55
MAX_Y_STD = 1.0
MAX_X_STD = 6.0
CLOSE_TTC_BLOCK_S = 4.0
HIGH_REQUIRED_DECEL_BLOCK = 0.35
MIN_CLOSING_SPEED = 0.15
MAX_MODEL_DISTANCE_MISMATCH = 8.0


@dataclass(frozen=True)
class LeadPathClearanceResult:
  requested_mode: str = MODE_OFF
  effective_mode: str = MODE_OFF
  enabled: bool = False
  apply_supported: bool = False
  shadow_eligible: bool = False
  shadow_blocked_reason: str = "mode_off"
  lead_idx: int = -1
  path_y_rel: float = 0.0
  lateral_velocity: float = 0.0
  t_clear: float = math.inf
  t_conflict: float = math.inf
  confidence: float = 0.0
  model_prob: float = 0.0
  ttc: float = math.inf
  required_decel: float = 0.0
  clear_abs_path_y: float = 0.0

  def debug_dict(self) -> dict[str, object]:
    prefix = "lead_path_clearance"
    return {
      f"{prefix}_mode": self.requested_mode,
      f"{prefix}_effective_mode": self.effective_mode,
      f"{prefix}_enabled": self.enabled,
      f"{prefix}_apply_supported": self.apply_supported,
      f"{prefix}_shadow_eligible": self.shadow_eligible,
      f"{prefix}_shadow_blocked_reason": self.shadow_blocked_reason,
      f"{prefix}_lead_idx": int(self.lead_idx),
      f"{prefix}_path_y_rel": float(self.path_y_rel),
      f"{prefix}_lateral_velocity": float(self.lateral_velocity),
      f"{prefix}_t_clear": _debug_time(self.t_clear),
      f"{prefix}_t_conflict": _debug_time(self.t_conflict),
      f"{prefix}_confidence": float(self.confidence),
      f"{prefix}_model_prob": float(self.model_prob),
      f"{prefix}_ttc": _debug_time(self.ttc),
      f"{prefix}_required_decel": float(self.required_decel),
      f"{prefix}_clear_abs_path_y": float(self.clear_abs_path_y),
    }


def normalize_mode(value: Any) -> str:
  if isinstance(value, bytes):
    value = value.decode(errors="ignore")
  text = str(value or "").strip().lower()
  return text if text in VALID_MODES else MODE_OFF


def predict_lead_path_clearance(mode: Any, lead_context: Any | None, model_msg: Any | None,
                                v_ego: float) -> LeadPathClearanceResult:
  requested_mode = normalize_mode(mode)
  effective_mode = MODE_SHADOW if requested_mode == MODE_APPLY else requested_mode
  base: dict[str, Any] = {
    "requested_mode": requested_mode,
    "effective_mode": effective_mode,
    "enabled": effective_mode == MODE_SHADOW,
    "apply_supported": False,
  }
  if effective_mode == MODE_OFF:
    return LeadPathClearanceResult(**base, shadow_blocked_reason="mode_off")
  if lead_context is None:
    return LeadPathClearanceResult(**base, shadow_blocked_reason="no_path_context")
  if bool(getattr(lead_context, "alternate_threat_active", False)):
    return LeadPathClearanceResult(**base, shadow_blocked_reason="alternate_lead_threat")

  state = _primary_state(lead_context)
  if state is None or not bool(getattr(state, "status", False)):
    return LeadPathClearanceResult(**base, shadow_blocked_reason="no_lead")
  if bool(getattr(state, "shadow", False)):
    return LeadPathClearanceResult(**base, shadow_blocked_reason="shadow_lead")

  lead_idx = _int(getattr(state, "lead_idx", -1), -1)
  path_y_rel = _f(getattr(state, "path_y_rel", 0.0))
  model_prob = _f(getattr(state, "model_prob", 0.0))
  ttc = _f(getattr(state, "ttc", math.inf), math.inf)
  required_decel = _f(getattr(state, "required_decel", 0.0))
  v_rel = _f(getattr(state, "v_rel", 0.0))
  common: dict[str, Any] = {
    **base,
    "lead_idx": lead_idx,
    "path_y_rel": path_y_rel,
    "model_prob": model_prob,
    "ttc": ttc,
    "required_decel": required_decel,
  }

  if not bool(getattr(state, "stable", False)) or bool(getattr(state, "new_lead", False)) or _f(getattr(state, "flicker_guard_timer", 0.0)) > 0.0:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="lead_not_stable")
  if _f(getattr(state, "confidence", 0.0)) < MIN_STATE_CONFIDENCE:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="low_lead_confidence")
  if math.isfinite(ttc) and ttc < CLOSE_TTC_BLOCK_S:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="close_ttc")
  if required_decel >= HIGH_REQUIRED_DECEL_BLOCK:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="high_required_decel")
  if -v_rel < MIN_CLOSING_SPEED:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="not_closing")

  traj = _model_lead_path_trajectory(model_msg, lead_idx)
  if traj is None:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="no_model_lead_trajectory")
  times, xs, path_ys, model_path_prob, x_stds, y_stds = traj
  model_prob = max(model_prob, model_path_prob)
  common["model_prob"] = model_prob
  if model_prob < MIN_MODEL_PROB:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="low_model_prob")
  if abs(xs[0] - _f(getattr(state, "d_rel", xs[0]))) > MAX_MODEL_DISTANCE_MISMATCH:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="model_distance_disagreement")
  if abs(path_ys[0] - path_y_rel) > 0.8 or (abs(path_y_rel) > 0.2 and path_ys[0] * path_y_rel < -0.05):
    return LeadPathClearanceResult(**common, shadow_blocked_reason="model_lateral_disagreement")
  if any(std > MAX_X_STD for std in x_stds) or any(std > MAX_Y_STD for std in y_stds):
    return LeadPathClearanceResult(**common, shadow_blocked_reason="model_uncertain")

  clear_idx = next((i for i, y in enumerate(path_ys) if abs(y) >= PATH_CLEAR_Y), None)
  if clear_idx is None:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="not_clearing_path", clear_abs_path_y=max(abs(y) for y in path_ys))
  if any(abs(y) < PATH_CLEAR_Y - PATH_CLEAR_SUSTAIN_TOL for y in path_ys[clear_idx:]):
    return LeadPathClearanceResult(**common, shadow_blocked_reason="clear_not_sustained", clear_abs_path_y=abs(path_ys[clear_idx]))

  t_clear = times[clear_idx]
  lateral_velocity = _lateral_velocity(times, path_ys, clear_idx)
  if not _moving_out(path_ys, clear_idx):
    return LeadPathClearanceResult(**common, shadow_blocked_reason="lead_not_moving_out", lateral_velocity=lateral_velocity,
                                   t_clear=t_clear, clear_abs_path_y=abs(path_ys[clear_idx]))

  t_conflict = ttc if math.isfinite(ttc) else _f(getattr(state, "d_rel", 0.0)) / max(_f(v_ego), 0.1)
  if not math.isfinite(t_conflict):
    return LeadPathClearanceResult(**common, shadow_blocked_reason="no_conflict_time", lateral_velocity=lateral_velocity,
                                   t_clear=t_clear, t_conflict=t_conflict, clear_abs_path_y=abs(path_ys[clear_idx]))
  if t_clear + MIN_CLEAR_MARGIN_S >= t_conflict:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="clears_too_late", lateral_velocity=lateral_velocity,
                                   t_clear=t_clear, t_conflict=t_conflict, clear_abs_path_y=abs(path_ys[clear_idx]))

  confidence = _clearance_confidence(state, model_prob, x_stds, y_stds, path_ys, clear_idx)
  if confidence < MIN_STATE_CONFIDENCE:
    return LeadPathClearanceResult(**common, shadow_blocked_reason="low_clearance_confidence", lateral_velocity=lateral_velocity,
                                   t_clear=t_clear, t_conflict=t_conflict, confidence=confidence,
                                   clear_abs_path_y=abs(path_ys[clear_idx]))
  return LeadPathClearanceResult(**common, shadow_eligible=True, shadow_blocked_reason="",
                                 lateral_velocity=lateral_velocity, t_clear=t_clear, t_conflict=t_conflict,
                                 confidence=confidence, clear_abs_path_y=abs(path_ys[clear_idx]))


def _primary_state(lead_context: Any) -> Any | None:
  primary = getattr(lead_context, "behavior", None) or getattr(lead_context, "physical", None)
  if primary is not None and bool(getattr(primary, "status", False)):
    return primary
  states = tuple(getattr(lead_context, "states", ()) or ())
  real_states = [s for s in states if bool(getattr(s, "status", False)) and not bool(getattr(s, "shadow", False))]
  return min(real_states, key=lambda s: _f(getattr(s, "d_rel", math.inf), math.inf), default=None)


def _model_lead_path_trajectory(model_msg: Any | None, lead_idx: int) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float, tuple[float, ...], tuple[float, ...]] | None:
  if model_msg is None or lead_idx < 0:
    return None
  leads = getattr(model_msg, "leadsV3", None)
  if leads is None or lead_idx >= len(leads):
    return None
  lead = leads[lead_idx]
  xs = tuple(_finite_sequence(getattr(lead, "x", ())))
  model_ys = tuple(_finite_sequence(getattr(lead, "y", ())))
  if len(xs) < 2 or len(xs) != len(model_ys):
    return None
  times = tuple(_finite_sequence(getattr(lead, "t", ())))
  if len(times) != len(xs):
    times = tuple(float(i) for i in range(len(xs)))
  # modelV2.leadsV3.y is opposite the planner/radarState yRel convention.
  planner_ys = tuple(-y for y in model_ys)
  path_ys = tuple(_path_relative_y(y, x, model_msg) for x, y in zip(xs, planner_ys, strict=True))
  if any(not math.isfinite(v) for v in (*times, *xs, *path_ys)):
    return None
  prob = _f(getattr(lead, "prob", 0.0))
  x_stds = _std_sequence(getattr(lead, "xStd", ()), len(xs))
  y_stds = _std_sequence(getattr(lead, "yStd", ()), len(xs))
  return times, xs, path_ys, prob, x_stds, y_stds


def _path_relative_y(y_rel: float, d_rel: float, model_msg: Any) -> float:
  path_y = _path_y_at(d_rel, model_msg)
  return y_rel if path_y is None else y_rel - path_y


def _path_y_at(d_rel: float, model_msg: Any) -> float | None:
  position = getattr(model_msg, "position", None)
  xs = tuple(_finite_sequence(getattr(position, "x", ()))) if position is not None else ()
  ys = tuple(_finite_sequence(getattr(position, "y", ()))) if position is not None else ()
  if len(xs) < 2 or len(xs) != len(ys) or d_rel < xs[0] or d_rel > xs[-1]:
    return None
  for i in range(len(xs) - 1):
    x0, x1 = xs[i], xs[i + 1]
    if x0 <= d_rel <= x1:
      if x1 == x0:
        return ys[i + 1]
      r = (d_rel - x0) / (x1 - x0)
      return ys[i] + r * (ys[i + 1] - ys[i])
  return None


def _moving_out(path_ys: tuple[float, ...], clear_idx: int) -> bool:
  direction = _direction(path_ys)
  if direction == 0.0:
    return False
  if any(y * direction < -0.05 for y in path_ys[:clear_idx + 1]):
    return False
  abs_ys = [abs(y) for y in path_ys[:clear_idx + 1]]
  if abs_ys[-1] < abs_ys[0] + MIN_LATERAL_DELTA:
    return False
  return all(abs_ys[i + 1] >= abs_ys[i] - 0.15 for i in range(len(abs_ys) - 1))


def _direction(path_ys: tuple[float, ...]) -> float:
  for y in path_ys:
    if abs(y) > 0.15:
      return math.copysign(1.0, y)
  return 0.0


def _lateral_velocity(times: tuple[float, ...], path_ys: tuple[float, ...], clear_idx: int) -> float:
  dt = max(times[clear_idx] - times[0], 0.1)
  return (path_ys[clear_idx] - path_ys[0]) / dt


def _clearance_confidence(state: Any, model_prob: float, x_stds: tuple[float, ...], y_stds: tuple[float, ...],
                          path_ys: tuple[float, ...], clear_idx: int) -> float:
  state_conf = _f(getattr(state, "confidence", 0.0))
  std_penalty = max(max(x_stds, default=0.0) / MAX_X_STD, max(y_stds, default=0.0) / MAX_Y_STD) * 0.20
  trend = min(1.0, max(0.0, (abs(path_ys[clear_idx]) - abs(path_ys[0])) / max(PATH_CLEAR_Y, 0.1)))
  return _clip(0.45 * state_conf + 0.35 * model_prob + 0.20 * trend - std_penalty)


def _std_sequence(value: Any, n: int) -> tuple[float, ...]:
  seq = tuple(_finite_sequence(value))
  if len(seq) == n:
    return seq
  scalar = _f(value, math.inf)
  if math.isfinite(scalar):
    return tuple(scalar for _ in range(n))
  return tuple(math.inf for _ in range(n))


def _finite_sequence(value: Any) -> list[float]:
  out: list[float] = []
  try:
    iterator = iter(value)
  except TypeError:
    v = _f(value, math.nan)
    return [v] if math.isfinite(v) else []
  for item in iterator:
    v = _f(item, math.nan)
    if not math.isfinite(v):
      return []
    out.append(v)
  return out


def _debug_time(value: float) -> float:
  value = _f(value, math.inf)
  return 0.0 if math.isinf(value) else float(value)


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
  return max(lower, min(upper, value))


def _f(value: Any, default: float = 0.0) -> float:
  try:
    v = float(value)
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


def _int(value: Any, default: int = 0) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default
