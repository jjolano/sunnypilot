from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_confidence import LEAD_CONFIDENCE_TRACK_UNKNOWN, LeadConfidenceState


LEAD_AUTHORITY_NONE = "none"
LEAD_AUTHORITY_SUPPRESS_ONLY = "suppress_only"
LEAD_AUTHORITY_PHYSICAL = "physical"
LEAD_AUTHORITY_PROGRESS_ALLOWED = "progress_allowed"

LEAD_CONTEXT_STOP_DISTANCE = 5.0
LEAD_CONTEXT_ON_PATH_Y = 0.6
LEAD_CONTEXT_PATH_EXIT_Y = 1.6
LEAD_CONTEXT_FALSE_POSITIVE_HOLD = 0.25
LEAD_CONTEXT_CLOSE_TIME_GAP = 2.2
LEAD_CONTEXT_CLOSE_DISTANCE = 25.0
LEAD_CONTEXT_CLOSE_STOP_D_REL = 15.0
LEAD_CONTEXT_CLOSE_STOP_V = 5.0
LEAD_CONTEXT_CLOSE_STOP_PULLAWAY_MIN_OPENING = 0.15
LEAD_CONTEXT_CLOSE_STOP_PULLAWAY_MIN_V_LEAD = 0.2
# Creep arms at the desired gap (1.0 m beyond the stop distance), not well beyond it — hypermile
# launch tuning (ADR 2026-06-13-longitudinal-hypermile-tuning §1).
LEAD_CONTEXT_STOP_GAP_CREEP_ARM_EXCESS = 1.0
LEAD_CONTEXT_STOP_GAP_CREEP_MAX_EXCESS = 1.25
LEAD_CONTEXT_STOP_GAP_CREEP_MIN_V_REL = -0.25
LEAD_CONTEXT_NEW_FAR_Y_REL = 1.6
LEAD_CONTEXT_NEW_FAR_MODEL_PROB = 0.5
LEAD_CONTEXT_NEW_FAR_REQUIRED_DECEL = 0.15
LEAD_CONTEXT_RISK_REQUIRED_DECEL = 0.25
LEAD_CONTEXT_RISK_TTC = 4.0
LEAD_CONTEXT_SHADOW_NORMAL_TIME = 0.4
LEAD_CONTEXT_SHADOW_RISK_TIME = 1.0
LEAD_CONTEXT_SHADOW_STOP_GO_TIME = 1.5
LEAD_CONTEXT_SHADOW_CUTOUT_EXIT_Y_REL = 1.2  # near-lane-edge threshold for plausible occlusion
LEAD_CONTEXT_SHADOW_CUTOUT_OUTWARD_TICKS = 2
LEAD_CONTEXT_SHADOW_CUTOUT_STABLE_TIME = 0.35
LEAD_CONTEXT_SHADOW_RECENT_CONSTRAIN_TIME = 1.0
LEAD_CONTEXT_PREVIEW_T = (0.0, 0.2, 0.6, 1.0)
LEAD_CONTEXT_DUPLICATE_D_REL_TOL = 0.15
LEAD_CONTEXT_DUPLICATE_V_TOL = 0.05
LEAD_CONTEXT_DUPLICATE_Y_TOL = 0.05
LEAD_CONTEXT_SWITCH_MIN_DWELL_S = 0.30
LEAD_CONTEXT_SWITCH_SCORE_MARGIN = 0.35
LEAD_CONTEXT_SWITCH_REQUIRED_DECEL_MARGIN = 0.15
LEAD_CONTEXT_SWITCH_TTC_MARGIN_S = 0.75
LEAD_CONTEXT_SWITCH_IMMEDIATE_TTC_S = 3.0
LEAD_CONTEXT_REPLACEMENT_EXIT_Y = 1.0
LEAD_CONTEXT_REPLACEMENT_ON_PATH_MIN = 0.5
LEAD_CONTEXT_REPLACEMENT_CONFIDENCE_MIN = 0.55
LEAD_CONTEXT_REPLACEMENT_MODEL_PROB_MIN = 0.5
LEAD_CONTEXT_REPLACEMENT_TTC = 5.0
LEAD_CONTEXT_REPLACEMENT_REQUIRED_DECEL = 0.20
LEAD_CONTEXT_REPLACEMENT_DISTANCE = 45.0


@dataclass(frozen=True)
class LeadTrajectoryPrediction:
  x: tuple[float, ...]
  v: tuple[float, ...]
  a: tuple[float, ...]
  valid: bool = False


@dataclass(frozen=True)
class LeadRiskModel:
  required_decel: float = 0.0
  ttc: float = math.inf
  time_gap: float = math.inf
  gap_shortage: float = 0.0
  closing_speed: float = 0.0
  stopped_or_crawling: bool = False
  path_y_rel: float = 0.0
  on_path_score: float = 0.0
  track_continuity: float = 0.0
  model_prob: float = 0.0
  radar_valid: bool = False
  ghost_score: float = 1.0


@dataclass(frozen=True)
class LeadProgressModel:
  opening_speed: float = 0.0
  lead_moving: bool = False
  lead_accel: float = 0.0
  predicted_gap_opening: bool = False
  gap_excess: float = 0.0
  stop_threat_absent: bool = False
  alternate_threat_absent: bool = True
  shadow_absent: bool = True
  confidence_stability_sufficient: bool = False
  allowed: bool = False
  reason: str = "no_progress_evidence"


@dataclass(frozen=True)
class LeadReplacementCandidate:
  active: bool = False
  replacing_idx: int = -1
  candidate_idx: int = -1
  score: float = 0.0
  reason: str = ""
  block_reason: str = ""


@dataclass(frozen=True)
class LeadRelevanceState:
  lead_idx: int
  status: bool
  shadow: bool
  stable: bool
  new_lead: bool
  flicker_guard_timer: float
  track_id: int
  d_rel: float
  y_rel: float
  path_y_rel: float
  v_lead: float
  v_rel: float
  model_prob: float
  radar: bool
  ttc: float
  required_decel: float
  time_gap: float
  on_path_score: float
  risk_score: float
  ghost_score: float
  confidence: float
  authority: str
  reason: str
  shadow_reason: str = ""
  shadow_occlusion_risk: float = 0.0
  shadow_path_y_rel_at_loss: float = 0.0
  shadow_stable_at_loss: bool = False
  shadow_age: float = 0.0
  shadow_duration: float = 0.0
  prediction: LeadTrajectoryPrediction = LeadTrajectoryPrediction((), (), (), False)
  risk_model: LeadRiskModel = field(default_factory=LeadRiskModel)
  progress_model: LeadProgressModel = field(default_factory=LeadProgressModel)

  @property
  def active(self) -> bool:
    return self.authority != LEAD_AUTHORITY_NONE

  @property
  def suppressive(self) -> bool:
    return self.authority in (LEAD_AUTHORITY_SUPPRESS_ONLY, LEAD_AUTHORITY_PHYSICAL, LEAD_AUTHORITY_PROGRESS_ALLOWED)

  @property
  def progress_allowed(self) -> bool:
    return self.authority == LEAD_AUTHORITY_PROGRESS_ALLOWED


@dataclass(frozen=True)
class PrimaryLeadContext:
  physical_idx: int | None
  behavior_idx: int | None
  physical: LeadRelevanceState | None
  behavior: LeadRelevanceState | None
  alternate_threat_active: bool
  shadow_active: bool
  reason: str
  states: tuple[LeadRelevanceState, ...] = ()
  lead_progress_allowed: bool = False
  lead_release_blocked_reason: str = ""
  replacement_candidate: LeadReplacementCandidate = field(default_factory=LeadReplacementCandidate)
  physical_switch_reason: str = ""
  physical_switch_dwell_s: float = 0.0
  physical_switch_prev_idx: int = -1
  physical_switch_prev_track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
  physical_switched: bool = False
  physical_switch_score_delta: float = 0.0

  @property
  def has_physical_lead(self) -> bool:
    return self.physical is not None and self.physical.suppressive

  def physical_lead_data(self, leads: tuple[Any, Any]) -> Any | None:
    return _lead_data_for_state(self.physical, leads)

  def behavior_lead_data(self, leads: tuple[Any, Any]) -> Any | None:
    return _lead_data_for_state(self.behavior, leads)

  @property
  def lead_gap_excess(self) -> float:
    # Surfaces the primary lead's progress-model gap excess so the stack can offer the
    # lead-pullaway progress candidate (policy gates it on gap_excess > 0). Without this the
    # stack's getattr fell back to 0.0 and lead-pullaway could never fire.
    primary = self.behavior or self.physical  # same primary the debug view uses
    return 0.0 if primary is None else float(primary.progress_model.gap_excess)

  def debug_dict(self) -> dict[str, object]:
    physical = self.physical
    behavior = self.behavior
    primary = behavior or physical
    active_shadow = next((state for state in self.states if state.shadow and state.suppressive), None)
    return {
      "primary_physical_lead_idx": -1 if self.physical_idx is None else int(self.physical_idx),
      "primary_behavior_lead_idx": -1 if self.behavior_idx is None else int(self.behavior_idx),
      "primary_lead_reason": str(self.reason),
      "primary_lead_authority": "" if primary is None else str(primary.authority),
      "alternate_lead_threat_active": bool(self.alternate_threat_active),
      "shadow_lead_active": bool(self.shadow_active),
      "lead_progress_allowed": bool(self.lead_progress_allowed),
      "lead_release_blocked_reason": str(self.lead_release_blocked_reason),
      "lead_replacement_active": bool(self.replacement_candidate.active),
      "lead_replacement_replacing_idx": int(self.replacement_candidate.replacing_idx),
      "lead_replacement_candidate_idx": int(self.replacement_candidate.candidate_idx),
      "lead_replacement_score": float(self.replacement_candidate.score),
      "lead_replacement_reason": str(self.replacement_candidate.reason),
      "lead_replacement_block_reason": str(self.replacement_candidate.block_reason),
      "primary_physical_switch_reason": str(self.physical_switch_reason),
      "primary_physical_switch_dwell_s": float(self.physical_switch_dwell_s),
      "primary_physical_prev_idx": int(self.physical_switch_prev_idx),
      "primary_physical_prev_track_id": int(self.physical_switch_prev_track_id),
      "primary_physical_switched": bool(self.physical_switched),
      "primary_physical_score_delta": float(self.physical_switch_score_delta),
      "primary_lead_d_rel": 0.0 if primary is None else float(primary.d_rel),
      "primary_lead_v_rel": 0.0 if primary is None else float(primary.v_rel),
      "primary_lead_y_rel": 0.0 if primary is None else float(primary.y_rel),
      "primary_lead_path_y_rel": 0.0 if primary is None else float(primary.path_y_rel),
      "primary_lead_risk_score": 0.0 if primary is None else float(primary.risk_score),
      "primary_lead_on_path_score": 0.0 if primary is None else float(primary.on_path_score),
      "primary_lead_required_decel": 0.0 if primary is None else float(primary.risk_model.required_decel),
      "primary_lead_ttc": 0.0 if primary is None or math.isinf(primary.risk_model.ttc) else float(primary.risk_model.ttc),
      "primary_lead_time_gap": 0.0 if primary is None or math.isinf(primary.risk_model.time_gap) else float(primary.risk_model.time_gap),
      "primary_lead_gap_shortage": 0.0 if primary is None else float(primary.risk_model.gap_shortage),
      "primary_lead_closing_speed": 0.0 if primary is None else float(primary.risk_model.closing_speed),
      "primary_lead_stopped_or_crawling": False if primary is None else bool(primary.risk_model.stopped_or_crawling),
      "primary_lead_ghost_score": 0.0 if primary is None else float(primary.risk_model.ghost_score),
      "primary_lead_progress_reason": "" if primary is None else str(primary.progress_model.reason),
      "primary_lead_progress_gap_excess": 0.0 if primary is None else float(primary.progress_model.gap_excess),
      "primary_lead_predicted_gap_opening": False if primary is None else bool(primary.progress_model.predicted_gap_opening),
      "shadow_lead_reason": "" if active_shadow is None else str(active_shadow.shadow_reason),
      "shadow_lead_duration": 0.0 if active_shadow is None else float(active_shadow.shadow_duration),
      "shadow_lead_age": 0.0 if active_shadow is None else float(active_shadow.shadow_age),
      "shadow_lead_occlusion_risk": 0.0 if active_shadow is None else float(active_shadow.shadow_occlusion_risk),
      "shadow_lead_path_y_rel_at_loss": 0.0 if active_shadow is None else float(active_shadow.shadow_path_y_rel_at_loss),
      "shadow_lead_stable_at_loss": False if active_shadow is None else bool(active_shadow.shadow_stable_at_loss),
    }


@dataclass(frozen=True)
class LeadShadowState:
  active: bool = False
  age: float = 0.0
  duration: float = 0.0
  lead_idx: int = -1
  track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
  d_rel: float = 0.0
  y_rel: float = 0.0
  v_lead: float = 0.0
  v_rel: float = 0.0
  a_lead: float = 0.0
  model_prob: float = 0.0
  radar: bool = False
  confidence: float = 0.0
  reason: str = ""
  occlusion_risk: float = 0.0
  path_y_rel_at_loss: float = 0.0
  stable_at_loss: bool = False


def _lead_data_for_state(state: LeadRelevanceState | None, leads: tuple[Any, Any]) -> Any | None:
  if state is None or state.shadow or state.lead_idx < 0 or state.lead_idx >= len(leads):
    return None
  lead = leads[state.lead_idx]
  return lead if bool(getattr(lead, "status", False)) else None


def finite_float(value: Any, default: float = 0.0) -> float:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return default
  return value if math.isfinite(value) else default


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
  return max(lower, min(upper, value))


def _lead_track_id(lead: Any) -> int:
  try:
    return int(getattr(lead, "radarTrackId", LEAD_CONFIDENCE_TRACK_UNKNOWN))
  except (TypeError, ValueError):
    return LEAD_CONFIDENCE_TRACK_UNKNOWN


def _path_relative_y(y_rel: float, d_rel: float, model_msg: Any | None) -> float:
  if model_msg is None:
    return y_rel
  positions = tuple(finite_float(x, math.nan) for x in getattr(getattr(model_msg, "position", None), "x", ()))
  path_y = tuple(finite_float(y, math.nan) for y in getattr(getattr(model_msg, "position", None), "y", ()))
  if len(positions) < 2 or len(positions) != len(path_y):
    return y_rel
  if any(not math.isfinite(value) for value in (*positions, *path_y, d_rel)):
    return y_rel
  if d_rel < positions[0] or d_rel > positions[-1]:
    return y_rel
  for idx in range(len(positions) - 1):
    x0, x1 = positions[idx], positions[idx + 1]
    if x0 <= d_rel <= x1:
      if x1 == x0:
        return y_rel - path_y[idx + 1]
      ratio = (d_rel - x0) / (x1 - x0)
      return y_rel - (path_y[idx] + ratio * (path_y[idx + 1] - path_y[idx]))
  return y_rel


A_LEAD_TAU_DEFAULT = 1.5  # s; lead-accel decay time constant (matches the MPC's aLeadTau use)


def lead_prediction(d_rel: float, v_lead: float, a_lead: float, v_ego: float, valid: bool = True,
                    a_lead_tau: float = A_LEAD_TAU_DEFAULT) -> LeadTrajectoryPrediction:
  # Decay the (noisy) lead accel toward zero with aLeadTau instead of propagating it constant,
  # as the MPC does — constant a_lead over the horizon was a likely lead-pullaway quirk source
  # (it makes predicted_gap_opening over-eager when the lead briefly accelerates).
  tau = max(finite_float(a_lead_tau, A_LEAD_TAU_DEFAULT), 0.1)
  xs: list[float] = []
  vs: list[float] = []
  accels: list[float] = []
  for t in LEAD_CONTEXT_PREVIEW_T:
    decay = math.exp(-t / tau)
    v = max(0.0, v_lead + a_lead * tau * (1.0 - decay))
    x = max(0.0, d_rel + (v_lead - v_ego) * t + a_lead * tau * (t - tau * (1.0 - decay)))
    xs.append(float(x))
    vs.append(float(v))
    accels.append(float(a_lead * decay))
  return LeadTrajectoryPrediction(tuple(xs), tuple(vs), tuple(accels), valid)


def _required_decel(d_rel: float, v_rel: float) -> float:
  closing_speed = max(0.0, -v_rel)
  return closing_speed * closing_speed / (2.0 * max(d_rel - LEAD_CONTEXT_STOP_DISTANCE, 0.1))


def _ttc(d_rel: float, v_rel: float) -> float:
  closing_speed = max(0.0, -v_rel)
  if closing_speed <= 0.05:
    return math.inf
  return max(0.0, d_rel) / closing_speed


def _time_gap(d_rel: float, v_ego: float) -> float:
  return max(0.0, d_rel) / max(v_ego, 0.1)


def _desired_progress_gap(v_ego: float) -> float:
  return max(LEAD_CONTEXT_CLOSE_DISTANCE, max(0.0, v_ego) * LEAD_CONTEXT_CLOSE_TIME_GAP)


def _gap_shortage(d_rel: float, v_ego: float) -> float:
  return max(0.0, _desired_progress_gap(v_ego) - max(0.0, d_rel))


def _gap_excess(d_rel: float, v_ego: float) -> float:
  return max(0.0, max(0.0, d_rel) - _desired_progress_gap(v_ego))


def _on_path_score(path_y_rel: float) -> float:
  abs_y = abs(path_y_rel)
  if abs_y <= LEAD_CONTEXT_ON_PATH_Y:
    return 1.0
  if abs_y >= LEAD_CONTEXT_PATH_EXIT_Y:
    return 0.0
  return 1.0 - (abs_y - LEAD_CONTEXT_ON_PATH_Y) / (LEAD_CONTEXT_PATH_EXIT_Y - LEAD_CONTEXT_ON_PATH_Y)


def _risk_score(d_rel: float, v_rel: float, v_lead: float, v_ego: float, required_decel: float, ttc: float, time_gap: float) -> float:
  closing = max(0.0, -v_rel, v_ego - v_lead)
  distance_risk = _clip((max(LEAD_CONTEXT_CLOSE_DISTANCE, v_ego * LEAD_CONTEXT_CLOSE_TIME_GAP) - d_rel) / LEAD_CONTEXT_CLOSE_DISTANCE)
  decel_risk = _clip(required_decel / 1.0)
  ttc_risk = 0.0 if math.isinf(ttc) else _clip((LEAD_CONTEXT_RISK_TTC - ttc) / LEAD_CONTEXT_RISK_TTC)
  time_gap_risk = _clip((LEAD_CONTEXT_CLOSE_TIME_GAP - time_gap) / LEAD_CONTEXT_CLOSE_TIME_GAP)
  stopped_risk = 0.45 if 0.0 < d_rel <= LEAD_CONTEXT_CLOSE_STOP_D_REL and 0.0 <= v_lead <= LEAD_CONTEXT_CLOSE_STOP_V else 0.0
  closing_risk = _clip(closing / 4.0)
  return max(decel_risk, ttc_risk, time_gap_risk, distance_risk * closing_risk, stopped_risk)


def _confidence_score(status: bool, shadow: bool, confidence_state: LeadConfidenceState | None, model_prob: float, radar: bool) -> float:
  if shadow:
    return 0.45
  if not status:
    return 0.0
  state = confidence_state or LeadConfidenceState(status=status)
  score = 0.15
  if state.stable:
    score += 0.45
  elif not state.new_lead:
    score += 0.25
  if state.speed_trusted:
    score += 0.15
  if radar:
    score += 0.10
  score += 0.20 * _clip(model_prob)
  return _clip(score)


def _ghost_score(on_path_score: float, risk_score: float, confidence: float, model_prob: float, radar: bool) -> float:
  weak_signal = 1.0 - max(confidence, 0.6 if radar else 0.0, model_prob)
  return _clip((1.0 - on_path_score) * 0.55 + weak_signal * 0.35 + (1.0 - risk_score) * 0.10)


def _is_close_or_closing(state: LeadRelevanceState) -> bool:
  return bool(
    state.required_decel >= LEAD_CONTEXT_RISK_REQUIRED_DECEL or
    state.risk_score >= 0.35 or
    state.ttc <= LEAD_CONTEXT_RISK_TTC or
    state.time_gap <= LEAD_CONTEXT_CLOSE_TIME_GAP
  )


def _lead_risk_model(required_decel: float, ttc: float, time_gap: float, d_rel: float, v_ego: float,
                     v_lead: float, v_rel: float, path_y_rel: float, on_path: float,
                     confidence_state: LeadConfidenceState | None, model_prob: float,
                     radar: bool, ghost: float) -> LeadRiskModel:
  track_continuity = 0.0
  if confidence_state is not None:
    track_continuity = 1.0 if confidence_state.stable else _clip(finite_float(confidence_state.age, 0.0))
  return LeadRiskModel(
    required_decel=required_decel,
    ttc=ttc,
    time_gap=time_gap,
    gap_shortage=_gap_shortage(d_rel, v_ego),
    closing_speed=max(0.0, -v_rel, v_ego - v_lead),
    stopped_or_crawling=0.0 <= v_lead <= LEAD_CONTEXT_CLOSE_STOP_V,
    path_y_rel=path_y_rel,
    on_path_score=on_path,
    track_continuity=track_continuity,
    model_prob=model_prob,
    radar_valid=bool(radar),
    ghost_score=ghost,
  )


def _lead_progress_model(d_rel: float, v_ego: float, v_lead: float, v_rel: float, a_lead: float,
                         on_path: float, confidence: float, confidence_state: LeadConfidenceState | None,
                         ghost: float, risk_model: LeadRiskModel, prediction: LeadTrajectoryPrediction,
                         shadow: bool = False, alternate_threat_absent: bool = True) -> LeadProgressModel:
  opening_speed = max(0.0, v_rel)
  lead_moving = v_lead > 0.2
  predicted_gap_opening = bool(prediction.valid and prediction.x and prediction.x[-1] > d_rel + 0.2)
  gap_excess = _gap_excess(d_rel, v_ego)
  closing_threat = bool(
    risk_model.required_decel >= LEAD_CONTEXT_RISK_REQUIRED_DECEL or
    risk_model.ttc <= LEAD_CONTEXT_RISK_TTC or
    risk_model.closing_speed > 0.05
  )
  close_gap_without_opening = risk_model.gap_shortage > 0.0 and opening_speed <= 0.15 and not predicted_gap_opening
  stopped_without_pullaway = risk_model.stopped_or_crawling and opening_speed <= 0.15 and not predicted_gap_opening
  stop_threat_absent = not closing_threat and not close_gap_without_opening and not stopped_without_pullaway
  stable = bool(confidence_state is not None and confidence_state.stable)
  new_or_flicker = bool(
    confidence_state is not None and (
      confidence_state.new_lead or confidence_state.guard_timer > 0.0 or confidence_state.flicker_guard_timer > 0.0
    )
  )
  confidence_stability_sufficient = bool(stable and confidence >= 0.55 and not new_or_flicker and on_path > 0.0 and ghost < 0.8)
  opening_evidence = opening_speed > 0.15 or (lead_moving and opening_speed > 0.05) or (a_lead > 0.10 and predicted_gap_opening)
  gap_excess_evidence = gap_excess > 0.0 and lead_moving
  allowed = bool(
    confidence_stability_sufficient and stop_threat_absent and alternate_threat_absent and not shadow and
    (opening_evidence or gap_excess_evidence)
  )
  if allowed:
    reason = "opening_or_gap_progress"
  elif not confidence_stability_sufficient:
    reason = "insufficient_confidence_stability"
  elif not stop_threat_absent:
    reason = "stop_or_closing_threat"
  elif shadow:
    reason = "shadow_no_progress"
  elif not alternate_threat_absent:
    reason = "alternate_threat"
  else:
    reason = "no_opening_or_gap_evidence"
  return LeadProgressModel(
    opening_speed=opening_speed,
    lead_moving=lead_moving,
    lead_accel=a_lead,
    predicted_gap_opening=predicted_gap_opening,
    gap_excess=gap_excess,
    stop_threat_absent=stop_threat_absent,
    alternate_threat_absent=alternate_threat_absent,
    shadow_absent=not shadow,
    confidence_stability_sufficient=confidence_stability_sufficient,
    allowed=allowed,
    reason=reason,
  )


class LeadShadowTracker:
  def __init__(self, lead_idx: int):
    self.lead_idx = int(lead_idx)
    self._shadow = LeadShadowState(lead_idx=self.lead_idx)
    self._last_real = LeadShadowState(lead_idx=self.lead_idx)
    self._was_status = False
    self._prev_path_y_rel = 0.0
    self._outward_tick_count = 0
    self._stable_at_loss = False
    self._constraining_timer = 0.0

  def reset(self) -> LeadShadowState:
    self._shadow = LeadShadowState(lead_idx=self.lead_idx)
    self._last_real = LeadShadowState(lead_idx=self.lead_idx)
    self._was_status = False
    self._prev_path_y_rel = 0.0
    self._outward_tick_count = 0
    self._stable_at_loss = False
    self._constraining_timer = 0.0
    return self._shadow

  def update(self, lead: Any, confidence_state: LeadConfidenceState | None, v_ego: float, dt: float,
             path_y_rel: float = 0.0, reset_state: bool = False) -> LeadShadowState:
    dt = max(0.0, finite_float(dt))
    status = bool(getattr(lead, "status", False)) if lead is not None else False
    if reset_state:
      return self.reset()

    if status:
      path_y_rel = finite_float(path_y_rel)
      self._update_real_history(lead, confidence_state, v_ego, path_y_rel, dt)
      self._last_real = self._snapshot_from_lead(lead, confidence_state, v_ego)
      self._shadow = LeadShadowState(lead_idx=self.lead_idx)
      self._was_status = True
      return self._shadow

    if self._was_status and not self._shadow.active:
      self._shadow = self._start_shadow(self._last_real, path_y_rel)
    self._was_status = False
    if not self._shadow.active:
      self._constraining_timer = max(0.0, self._constraining_timer - dt)
      return self._shadow

    age = self._shadow.age + dt
    v_lead = max(0.0, self._shadow.v_lead + self._shadow.a_lead * dt)
    v_rel = v_lead - v_ego
    d_rel = max(0.0, self._shadow.d_rel + v_rel * dt)
    confidence = _clip(self._shadow.confidence * max(0.0, 1.0 - age / max(self._shadow.duration, 0.1)))
    active = bool(age <= self._shadow.duration and d_rel > 0.0)
    self._shadow = LeadShadowState(
      active=active,
      age=age,
      duration=self._shadow.duration,
      lead_idx=self.lead_idx,
      track_id=self._shadow.track_id,
      d_rel=d_rel,
      y_rel=self._shadow.y_rel,
      v_lead=v_lead,
      v_rel=v_rel,
      a_lead=self._shadow.a_lead,
      model_prob=self._shadow.model_prob,
      radar=self._shadow.radar,
      confidence=confidence,
      reason=self._shadow.reason,
      occlusion_risk=self._shadow.occlusion_risk,
      path_y_rel_at_loss=self._shadow.path_y_rel_at_loss,
      stable_at_loss=self._shadow.stable_at_loss,
    )
    return self._shadow

  def _update_real_history(self, lead: Any, confidence_state: LeadConfidenceState | None,
                           v_ego: float, path_y_rel: float, dt: float) -> None:
    if confidence_state is not None:
      self._stable_at_loss = bool(
        confidence_state.stable or confidence_state.age >= LEAD_CONTEXT_SHADOW_CUTOUT_STABLE_TIME
      )
    else:
      self._stable_at_loss = False

    d_rel = finite_float(getattr(lead, "dRel", 0.0))
    v_lead = finite_float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0)))
    v_rel = finite_float(getattr(lead, "vRel", v_lead - v_ego), v_lead - v_ego)
    required_decel = _required_decel(d_rel, v_rel)
    ttc = _ttc(d_rel, v_rel)
    time_gap = _time_gap(d_rel, v_ego)
    risk = _risk_score(d_rel, v_rel, v_lead, v_ego, required_decel, ttc, time_gap)
    constraining = bool(
      required_decel >= LEAD_CONTEXT_RISK_REQUIRED_DECEL or
      ttc <= LEAD_CONTEXT_RISK_TTC or
      risk >= 0.35
    )
    if constraining:
      self._constraining_timer = LEAD_CONTEXT_SHADOW_RECENT_CONSTRAIN_TIME
    else:
      self._constraining_timer = max(0.0, self._constraining_timer - dt)

    if self._was_status:
      outward = abs(path_y_rel) > abs(self._prev_path_y_rel) + 0.02
      if outward and abs(path_y_rel) >= LEAD_CONTEXT_ON_PATH_Y:
        self._outward_tick_count = min(self._outward_tick_count + 1, 10)
      else:
        self._outward_tick_count = 0
    else:
      self._outward_tick_count = 0
    self._prev_path_y_rel = path_y_rel

  def _snapshot_from_lead(self, lead: Any, confidence_state: LeadConfidenceState | None, v_ego: float) -> LeadShadowState:
    v_lead = finite_float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0)))
    v_rel = finite_float(getattr(lead, "vRel", v_lead - v_ego), v_lead - v_ego)
    model_prob = finite_float(getattr(lead, "modelProb", 0.0))
    radar = bool(getattr(lead, "radar", False))
    return LeadShadowState(
      active=False,
      lead_idx=self.lead_idx,
      track_id=_lead_track_id(lead),
      d_rel=finite_float(getattr(lead, "dRel", 0.0)),
      y_rel=finite_float(getattr(lead, "yRel", 0.0)),
      v_lead=v_lead,
      v_rel=v_rel,
      a_lead=finite_float(getattr(lead, "aLeadK", 0.0)),
      model_prob=model_prob,
      radar=radar,
      confidence=_confidence_score(True, False, confidence_state, model_prob, radar),
    )

  def _start_shadow(self, last: LeadShadowState, path_y_rel: float) -> LeadShadowState:
    required_decel = _required_decel(last.d_rel, last.v_rel)
    ttc = _ttc(last.d_rel, last.v_rel)
    time_gap = math.inf
    risk = _risk_score(last.d_rel, last.v_rel, last.v_lead, last.v_lead - last.v_rel, required_decel, ttc, time_gap)
    close_stop_go = 0.0 < last.d_rel <= LEAD_CONTEXT_CLOSE_STOP_D_REL and 0.0 <= last.v_lead <= LEAD_CONTEXT_CLOSE_STOP_V
    close_or_closing = bool(required_decel >= LEAD_CONTEXT_RISK_REQUIRED_DECEL or ttc <= LEAD_CONTEXT_RISK_TTC or risk >= 0.35)

    v_ego_at_loss = max(0.1, last.v_lead - last.v_rel)
    cutout_time_gap = _time_gap(last.d_rel, v_ego_at_loss)
    path_y_rel = finite_float(path_y_rel)
    near_exit = abs(path_y_rel) >= LEAD_CONTEXT_SHADOW_CUTOUT_EXIT_Y_REL
    outward_exit = bool(
      self._outward_tick_count >= LEAD_CONTEXT_SHADOW_CUTOUT_OUTWARD_TICKS and
      abs(path_y_rel) >= LEAD_CONTEXT_ON_PATH_Y
    )
    lateral_exit = bool(near_exit or outward_exit or abs(path_y_rel) >= LEAD_CONTEXT_PATH_EXIT_Y)
    occlusive = bool(
      close_or_closing or
      close_stop_go or
      last.d_rel <= LEAD_CONTEXT_CLOSE_DISTANCE or
      cutout_time_gap <= LEAD_CONTEXT_CLOSE_TIME_GAP or
      last.v_lead <= LEAD_CONTEXT_CLOSE_STOP_V or
      last.a_lead <= -1.5 or
      self._constraining_timer > 0.0
    )
    cutout_exit = bool(self._stable_at_loss and occlusive and lateral_exit)

    if close_stop_go:
      duration = LEAD_CONTEXT_SHADOW_STOP_GO_TIME
      reason = "stop_go_dropout"
      occlusion_risk = 0.0
    elif cutout_exit:
      duration = LEAD_CONTEXT_SHADOW_RISK_TIME
      reason = "cutout_exit"
      occlusion_risk = 1.0
    elif close_or_closing:
      duration = LEAD_CONTEXT_SHADOW_RISK_TIME
      reason = "risk_dropout"
      occlusion_risk = 0.0
    else:
      duration = LEAD_CONTEXT_SHADOW_NORMAL_TIME
      reason = "dropout"
      occlusion_risk = 0.0
    return LeadShadowState(
      active=True,
      age=0.0,
      duration=duration,
      lead_idx=self.lead_idx,
      track_id=last.track_id,
      d_rel=last.d_rel,
      y_rel=last.y_rel,
      v_lead=last.v_lead,
      v_rel=last.v_rel,
      a_lead=last.a_lead,
      model_prob=last.model_prob,
      radar=last.radar,
      confidence=last.confidence,
      reason=reason,
      occlusion_risk=occlusion_risk,
      path_y_rel_at_loss=path_y_rel,
      stable_at_loss=self._stable_at_loss,
    )


class LeadContextTracker:
  def __init__(self):
    self.shadow_trackers = (LeadShadowTracker(0), LeadShadowTracker(1))
    self._false_positive_release_timers = [0.0, 0.0]
    self._prev_path_y_rel = [0.0, 0.0]
    self._prev_physical_idx: int | None = None
    self._prev_physical_track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN
    self._physical_dwell_s: float = 0.0

  def reset(self) -> None:
    for tracker in self.shadow_trackers:
      tracker.reset()
    self._false_positive_release_timers = [0.0, 0.0]
    self._prev_path_y_rel = [0.0, 0.0]
    self._prev_physical_idx = None
    self._prev_physical_track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
    self._physical_dwell_s = 0.0

  def update(self, leads: tuple[Any, Any], confidence_states: tuple[LeadConfidenceState, LeadConfidenceState] | list[LeadConfidenceState],
             v_ego: float, dt: float, model_msg: Any | None = None, dominant_idx: int | None = None,
             lead_dominant_idx: int | None = None, reset_state: bool = False) -> PrimaryLeadContext:
    dt = max(0.0, finite_float(dt))
    v_ego = finite_float(v_ego)
    states: list[LeadRelevanceState] = []
    for idx, lead in enumerate(leads):
      confidence_state = confidence_states[idx] if idx < len(confidence_states) else LeadConfidenceState()
      status = bool(getattr(lead, "status", False)) if lead is not None else False
      raw_y_rel = finite_float(getattr(lead, "yRel", 0.0)) if status else self._prev_path_y_rel[idx]
      raw_d_rel = finite_float(getattr(lead, "dRel", 0.0)) if status else 0.0
      path_y_rel = _path_relative_y(raw_y_rel, raw_d_rel, model_msg) if status else self._prev_path_y_rel[idx]
      shadow = self.shadow_trackers[idx].update(lead, confidence_state, v_ego, dt, path_y_rel, reset_state=reset_state)
      if status:
        state = self._real_state(idx, lead, confidence_state, v_ego, model_msg, dt)
      elif shadow.active:
        state = self._shadow_state(shadow, v_ego, model_msg)
      else:
        state = _empty_state(idx)
      states.append(state)

    states = [self._apply_false_positive_release(idx, state, dt) for idx, state in enumerate(states)]
    for idx, state in enumerate(states):
      if state.status:
        self._prev_path_y_rel[idx] = state.path_y_rel
    ctx = select_primary_lead_context(
      tuple(states), dominant_idx=dominant_idx, lead_dominant_idx=lead_dominant_idx,
      previous_physical_idx=self._prev_physical_idx,
      previous_physical_track_id=self._prev_physical_track_id,
      previous_physical_dwell_s=self._physical_dwell_s,
    )
    self._update_physical_memory(ctx, dt)
    return ctx

  def _update_physical_memory(self, ctx: PrimaryLeadContext, dt: float) -> None:
    physical = ctx.physical
    if physical is None or not physical.suppressive:
      self._prev_physical_idx = None
      self._prev_physical_track_id = LEAD_CONFIDENCE_TRACK_UNKNOWN
      self._physical_dwell_s = 0.0
      return
    same_idx = self._prev_physical_idx == physical.lead_idx
    same_track = bool(
      self._prev_physical_track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN and
      self._prev_physical_track_id == physical.track_id
    )
    if same_idx or same_track:
      self._physical_dwell_s += max(0.0, dt)
    else:
      self._physical_dwell_s = 0.0
    self._prev_physical_idx = physical.lead_idx
    self._prev_physical_track_id = physical.track_id

  def _real_state(self, idx: int, lead: Any, confidence_state: LeadConfidenceState, v_ego: float,
                  model_msg: Any | None, dt: float) -> LeadRelevanceState:
    d_rel = finite_float(getattr(lead, "dRel", 0.0))
    y_rel = finite_float(getattr(lead, "yRel", 0.0))
    path_y_rel = _path_relative_y(y_rel, d_rel, model_msg)
    v_lead = finite_float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0)))
    v_rel = finite_float(getattr(lead, "vRel", v_lead - v_ego), v_lead - v_ego)
    model_prob = finite_float(getattr(lead, "modelProb", 0.0))
    radar = bool(getattr(lead, "radar", False))
    required_decel = _required_decel(d_rel, v_rel)
    ttc = _ttc(d_rel, v_rel)
    time_gap = _time_gap(d_rel, v_ego)
    on_path = _on_path_score(path_y_rel)
    risk = _risk_score(d_rel, v_rel, v_lead, v_ego, required_decel, ttc, time_gap)
    confidence = _confidence_score(True, False, confidence_state, model_prob, radar)
    ghost = _ghost_score(on_path, risk, confidence, model_prob, radar)
    a_lead = finite_float(getattr(lead, "aLeadK", 0.0))
    a_lead_tau = finite_float(getattr(lead, "aLeadTau", A_LEAD_TAU_DEFAULT), A_LEAD_TAU_DEFAULT)
    prediction = lead_prediction(d_rel, v_lead, a_lead, v_ego, True, a_lead_tau)
    risk_model = _lead_risk_model(
      required_decel, ttc, time_gap, d_rel, v_ego, v_lead, v_rel, path_y_rel, on_path,
      confidence_state, model_prob, radar, ghost,
    )
    progress_model = _lead_progress_model(
      d_rel, v_ego, v_lead, v_rel, a_lead, on_path, confidence, confidence_state, ghost, risk_model, prediction,
    )
    authority, reason = self._authority_for_real_lead(idx, confidence_state, path_y_rel, on_path, risk, required_decel, ttc,
                                                      d_rel, v_lead, v_rel, time_gap, model_prob, confidence, ghost,
                                                      progress_model)
    return LeadRelevanceState(
      lead_idx=idx,
      status=True,
      shadow=False,
      stable=bool(confidence_state.stable),
      new_lead=bool(confidence_state.new_lead or confidence_state.guard_timer > 0.0),
      flicker_guard_timer=finite_float(confidence_state.flicker_guard_timer),
      track_id=_lead_track_id(lead),
      d_rel=d_rel,
      y_rel=y_rel,
      path_y_rel=path_y_rel,
      v_lead=v_lead,
      v_rel=v_rel,
      model_prob=model_prob,
      radar=radar,
      ttc=ttc,
      required_decel=required_decel,
      time_gap=time_gap,
      on_path_score=on_path,
      risk_score=risk,
      ghost_score=ghost,
      confidence=confidence,
      authority=authority,
      reason=reason,
      prediction=prediction,
      risk_model=risk_model,
      progress_model=progress_model,
    )

  def _shadow_state(self, shadow: LeadShadowState, v_ego: float, model_msg: Any | None) -> LeadRelevanceState:
    path_y_rel = _path_relative_y(shadow.y_rel, shadow.d_rel, model_msg)
    required_decel = _required_decel(shadow.d_rel, shadow.v_rel)
    ttc = _ttc(shadow.d_rel, shadow.v_rel)
    time_gap = _time_gap(shadow.d_rel, v_ego)
    on_path = _on_path_score(path_y_rel)
    risk = _risk_score(shadow.d_rel, shadow.v_rel, shadow.v_lead, v_ego, required_decel, ttc, time_gap)
    confidence = _clip(shadow.confidence)
    ghost = _ghost_score(on_path, risk, confidence, shadow.model_prob, shadow.radar)
    prediction = lead_prediction(shadow.d_rel, shadow.v_lead, shadow.a_lead, v_ego, True)
    risk_model = _lead_risk_model(
      required_decel, ttc, time_gap, shadow.d_rel, v_ego, shadow.v_lead, shadow.v_rel, path_y_rel, on_path,
      None, shadow.model_prob, shadow.radar, ghost,
    )
    progress_model = _lead_progress_model(
      shadow.d_rel, v_ego, shadow.v_lead, shadow.v_rel, shadow.a_lead, on_path, confidence, None,
      ghost, risk_model, prediction, shadow=True,
    )
    return LeadRelevanceState(
      lead_idx=shadow.lead_idx,
      status=False,
      shadow=True,
      stable=False,
      new_lead=False,
      flicker_guard_timer=max(0.0, shadow.duration - shadow.age),
      track_id=shadow.track_id,
      d_rel=shadow.d_rel,
      y_rel=shadow.y_rel,
      path_y_rel=path_y_rel,
      v_lead=shadow.v_lead,
      v_rel=shadow.v_rel,
      model_prob=shadow.model_prob,
      radar=shadow.radar,
      ttc=ttc,
      required_decel=required_decel,
      time_gap=time_gap,
      on_path_score=on_path,
      risk_score=risk,
      ghost_score=ghost,
      confidence=confidence,
      authority=LEAD_AUTHORITY_SUPPRESS_ONLY,
      reason=shadow.reason if shadow.reason else "shadow_lead_suppress_only",
      shadow_reason=shadow.reason,
      shadow_occlusion_risk=shadow.occlusion_risk,
      shadow_path_y_rel_at_loss=shadow.path_y_rel_at_loss,
      shadow_stable_at_loss=shadow.stable_at_loss,
      shadow_age=shadow.age,
      shadow_duration=shadow.duration,
      prediction=prediction,
      risk_model=risk_model,
      progress_model=progress_model,
    )

  def _authority_for_real_lead(self, idx: int, confidence_state: LeadConfidenceState, path_y_rel: float, on_path: float,
                               risk: float, required_decel: float, ttc: float, d_rel: float, v_lead: float,
                               v_rel: float, time_gap: float, model_prob: float, confidence: float,
                               ghost: float, progress_model: LeadProgressModel) -> tuple[str, str]:
    close_or_closing = bool(required_decel >= LEAD_CONTEXT_RISK_REQUIRED_DECEL or ttc <= LEAD_CONTEXT_RISK_TTC or risk >= 0.35)
    low_risk_path_exit = self._false_positive_release_timers[idx] >= LEAD_CONTEXT_FALSE_POSITIVE_HOLD
    if low_risk_path_exit and (not close_or_closing or abs(path_y_rel) >= LEAD_CONTEXT_PATH_EXIT_Y):
      return LEAD_AUTHORITY_NONE, "lateral_exit_confirmed"
    if abs(path_y_rel) >= LEAD_CONTEXT_NEW_FAR_Y_REL and not close_or_closing and model_prob < LEAD_CONTEXT_NEW_FAR_MODEL_PROB:
      return LEAD_AUTHORITY_NONE, "path_relevance_low"
    if confidence_state.flicker_guard_timer > 0.0:
      return LEAD_AUTHORITY_SUPPRESS_ONLY, "flicker_guard_suppress_only"
    if confidence_state.new_lead or confidence_state.guard_timer > 0.0:
      if close_or_closing or on_path > 0.0:
        return LEAD_AUTHORITY_SUPPRESS_ONLY, "new_lead_suppress_only"
      return LEAD_AUTHORITY_NONE, "new_low_relevance_lead"
    if on_path <= 0.0 and not close_or_closing:
      return LEAD_AUTHORITY_SUPPRESS_ONLY, "path_exit_pending_release"
    if _close_stop_pullaway_progress_allowed(d_rel, v_lead, v_rel, progress_model):
      return LEAD_AUTHORITY_PROGRESS_ALLOWED, "stable_close_stop_pullaway_authorized_lead"
    if _stopped_gap_creep_progress_allowed(d_rel, v_lead, v_rel, progress_model):
      return LEAD_AUTHORITY_PROGRESS_ALLOWED, "stable_stopped_gap_creep_authorized_lead"
    if close_or_closing:
      return LEAD_AUTHORITY_PHYSICAL, "close_or_closing_lead"
    if progress_model.allowed:
      return LEAD_AUTHORITY_PROGRESS_ALLOWED, "stable_progress_authorized_lead"
    if on_path > 0.0 and confidence >= 0.45:
      return LEAD_AUTHORITY_PHYSICAL, "path_relevant_physical_lead"
    return LEAD_AUTHORITY_NONE, "weak_lead_evidence"

  def _apply_false_positive_release(self, idx: int, state: LeadRelevanceState, dt: float) -> LeadRelevanceState:
    if not state.status or state.shadow:
      self._false_positive_release_timers[idx] = 0.0
      return state
    prev_abs_y = abs(self._prev_path_y_rel[idx])
    abs_y = abs(state.path_y_rel)
    moving_out = abs_y >= prev_abs_y - 0.02
    low_risk = state.required_decel < LEAD_CONTEXT_NEW_FAR_REQUIRED_DECEL and (math.isinf(state.ttc) or state.ttc > LEAD_CONTEXT_RISK_TTC)
    weak_signal = state.model_prob < 0.6 or state.confidence < 0.55
    previously_released = self._false_positive_release_timers[idx] >= LEAD_CONTEXT_FALSE_POSITIVE_HOLD
    release_evidence = abs_y >= LEAD_CONTEXT_PATH_EXIT_Y and moving_out and ((low_risk and weak_signal) or previously_released)
    if release_evidence:
      self._false_positive_release_timers[idx] += dt
    else:
      self._false_positive_release_timers[idx] = 0.0
    if self._false_positive_release_timers[idx] < LEAD_CONTEXT_FALSE_POSITIVE_HOLD:
      return state
    return LeadRelevanceState(
      **{**state.__dict__, "authority": LEAD_AUTHORITY_NONE, "reason": "lateral_exit_confirmed"}
    )


def _empty_state(idx: int) -> LeadRelevanceState:
  return LeadRelevanceState(
    lead_idx=idx,
    status=False,
    shadow=False,
    stable=False,
    new_lead=False,
    flicker_guard_timer=0.0,
    track_id=LEAD_CONFIDENCE_TRACK_UNKNOWN,
    d_rel=0.0,
    y_rel=0.0,
    path_y_rel=0.0,
    v_lead=0.0,
    v_rel=0.0,
    model_prob=0.0,
    radar=False,
    ttc=math.inf,
    required_decel=0.0,
    time_gap=math.inf,
    on_path_score=0.0,
    risk_score=0.0,
    ghost_score=1.0,
    confidence=0.0,
    authority=LEAD_AUTHORITY_NONE,
    reason="no_lead",
    prediction=LeadTrajectoryPrediction((), (), (), False),
  )


def select_primary_lead_context(states: tuple[LeadRelevanceState, ...], dominant_idx: int | None = None,
                                lead_dominant_idx: int | None = None,
                                previous_physical_idx: int | None = None,
                                previous_physical_track_id: int = LEAD_CONFIDENCE_TRACK_UNKNOWN,
                                previous_physical_dwell_s: float = 0.0) -> PrimaryLeadContext:
  raw_physical = max((state for state in states if state.suppressive), key=lambda state: _physical_rank(state, dominant_idx, lead_dominant_idx), default=None)
  physical, switch_reason, switched, switch_score_delta = _apply_physical_hysteresis(
    states, raw_physical, dominant_idx, lead_dominant_idx,
    previous_physical_idx, previous_physical_track_id, previous_physical_dwell_s,
  )
  behavior = max((state for state in states if state.progress_allowed), key=lambda state: _behavior_rank(state, lead_dominant_idx), default=None)
  replacement = _replacement_candidate(states, physical, behavior)
  alternate_threat_active = bool(any(_alternate_threat(state, behavior) for state in states) or replacement.active)
  physical_conflicts_with_behavior = bool(
    physical is not None and behavior is not None and physical.lead_idx != behavior.lead_idx and (
      physical.shadow or physical.authority != LEAD_AUTHORITY_PROGRESS_ALLOWED or _is_close_or_closing(physical) or physical.v_lead <= 0.2
    )
  )
  shadow_active = any(state.shadow and state.suppressive for state in states)
  lead_progress_allowed = bool(behavior is not None and not alternate_threat_active and not physical_conflicts_with_behavior)
  if behavior is None:
    blocked_reason = "no_behavior_lead" if physical is not None else ""
  elif replacement.active:
    blocked_reason = "replacement_threat"
  elif alternate_threat_active:
    blocked_reason = "alternate_lead_threat"
  elif physical_conflicts_with_behavior:
    blocked_reason = "primary_physical_lead_suppressive"
  else:
    blocked_reason = ""
  reason = _context_reason(physical, behavior, blocked_reason)
  return PrimaryLeadContext(
    physical_idx=None if physical is None else physical.lead_idx,
    behavior_idx=None if behavior is None else behavior.lead_idx,
    physical=physical,
    behavior=behavior,
    alternate_threat_active=alternate_threat_active,
    shadow_active=shadow_active,
    reason=reason,
    states=states,
    lead_progress_allowed=lead_progress_allowed,
    lead_release_blocked_reason=blocked_reason,
    replacement_candidate=replacement,
    physical_switch_reason=switch_reason,
    physical_switch_dwell_s=float(previous_physical_dwell_s),
    physical_switch_prev_idx=-1 if previous_physical_idx is None else int(previous_physical_idx),
    physical_switch_prev_track_id=int(previous_physical_track_id),
    physical_switched=switched,
    physical_switch_score_delta=switch_score_delta,
  )


def _physical_rank(state: LeadRelevanceState, dominant_idx: int | None, lead_dominant_idx: int | None) -> float:
  hint = 0.0
  if state.lead_idx == dominant_idx:
    hint += 1.0
  if state.lead_idx == lead_dominant_idx:
    hint += 0.6
  shadow_penalty = 0.4 if state.shadow else 0.0
  return 4.0 * state.risk_score + 1.5 * state.on_path_score + state.confidence + hint - state.ghost_score - shadow_penalty


def _apply_physical_hysteresis(states: tuple[LeadRelevanceState, ...], raw_physical: LeadRelevanceState | None,
                               dominant_idx: int | None, lead_dominant_idx: int | None,
                               previous_idx: int | None, previous_track_id: int,
                               previous_dwell_s: float) -> tuple[LeadRelevanceState | None, str, bool, float]:
  if raw_physical is None or previous_idx is None:
    return raw_physical, "no_previous" if raw_physical is not None else "no_physical", raw_physical is not None, 0.0
  previous = _previous_physical_state(states, previous_idx, previous_track_id)
  if previous is None or not previous.suppressive:
    return raw_physical, "previous_unavailable", raw_physical.lead_idx != previous_idx, 0.0
  if raw_physical.lead_idx == previous.lead_idx or _same_physical_identity(raw_physical, previous):
    return raw_physical, "same_physical", False, 0.0

  raw_score = _physical_rank(raw_physical, dominant_idx, lead_dominant_idx)
  previous_score = _physical_rank(previous, dominant_idx, lead_dominant_idx)
  score_delta = raw_score - previous_score
  immediate = bool(
    raw_physical.ttc <= LEAD_CONTEXT_SWITCH_IMMEDIATE_TTC_S or
    raw_physical.required_decel >= previous.required_decel + LEAD_CONTEXT_SWITCH_REQUIRED_DECEL_MARGIN or
    (not math.isinf(previous.ttc) and raw_physical.ttc <= previous.ttc - LEAD_CONTEXT_SWITCH_TTC_MARGIN_S)
  )
  previous_released = bool(
    previous.authority == LEAD_AUTHORITY_NONE or
    (previous.on_path_score <= 0.0 and not _is_close_or_closing(previous)) or
    previous.reason in ("lateral_exit_confirmed", "path_relevance_low")
  )
  if previous_released:
    return raw_physical, "previous_released", True, score_delta
  if immediate:
    return raw_physical, "immediate_threat", True, score_delta
  if previous_dwell_s < LEAD_CONTEXT_SWITCH_MIN_DWELL_S and score_delta < LEAD_CONTEXT_SWITCH_SCORE_MARGIN:
    return previous, "hysteresis_keep_previous", False, score_delta
  if previous_dwell_s >= LEAD_CONTEXT_SWITCH_MIN_DWELL_S:
    return raw_physical, "dwell_elapsed", True, score_delta
  if score_delta >= LEAD_CONTEXT_SWITCH_SCORE_MARGIN:
    return raw_physical, "score_margin", True, score_delta
  return previous, "hysteresis_keep_previous", False, score_delta


def _previous_physical_state(states: tuple[LeadRelevanceState, ...], previous_idx: int,
                             previous_track_id: int) -> LeadRelevanceState | None:
  if previous_track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN:
    for state in states:
      if state.track_id == previous_track_id and state.suppressive:
        return state
  for state in states:
    if state.lead_idx == previous_idx and state.suppressive:
      return state
  return None


def _same_physical_identity(a: LeadRelevanceState, b: LeadRelevanceState) -> bool:
  a_track_known = a.track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN
  b_track_known = b.track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN
  if a_track_known or b_track_known:
    return bool(a_track_known and b_track_known and a.track_id == b.track_id)
  return bool(
    abs(a.d_rel - b.d_rel) <= 1.0 and
    abs(a.v_lead - b.v_lead) <= 1.0 and
    abs(a.v_rel - b.v_rel) <= 1.0 and
    abs(a.path_y_rel - b.path_y_rel) <= 0.75
  )


def _replacement_candidate(states: tuple[LeadRelevanceState, ...], physical: LeadRelevanceState | None,
                           behavior: LeadRelevanceState | None) -> LeadReplacementCandidate:
  replacing = behavior or physical
  if replacing is None or replacing.shadow or not replacing.status:
    return LeadReplacementCandidate(reason="no_replacing_lead")
  exiting = bool(
    abs(replacing.path_y_rel) >= LEAD_CONTEXT_REPLACEMENT_EXIT_Y or
    replacing.on_path_score < LEAD_CONTEXT_REPLACEMENT_ON_PATH_MIN or
    replacing.reason in ("path_exit_pending_release", "lateral_exit_confirmed")
  )
  if not exiting:
    return LeadReplacementCandidate(replacing_idx=replacing.lead_idx, reason="replacing_not_exiting")

  best: LeadReplacementCandidate | None = None
  for candidate in states:
    if candidate.lead_idx == replacing.lead_idx or candidate.shadow or not candidate.status:
      continue
    if _same_lead_duplicate(candidate, replacing):
      continue
    if candidate.new_lead or candidate.flicker_guard_timer > 0.0:
      continue
    if candidate.on_path_score < LEAD_CONTEXT_REPLACEMENT_ON_PATH_MIN and abs(candidate.path_y_rel) > LEAD_CONTEXT_ON_PATH_Y:
      continue
    credible = bool(
      (candidate.stable and candidate.confidence >= LEAD_CONTEXT_REPLACEMENT_CONFIDENCE_MIN) or
      (candidate.radar and candidate.model_prob >= LEAD_CONTEXT_REPLACEMENT_MODEL_PROB_MIN)
    )
    if not credible:
      continue
    threat = bool(
      candidate.ttc <= LEAD_CONTEXT_REPLACEMENT_TTC or
      candidate.required_decel >= LEAD_CONTEXT_REPLACEMENT_REQUIRED_DECEL or
      candidate.time_gap <= LEAD_CONTEXT_CLOSE_TIME_GAP
    )
    if not threat:
      continue
    if candidate.d_rel > LEAD_CONTEXT_REPLACEMENT_DISTANCE and candidate.time_gap > LEAD_CONTEXT_CLOSE_TIME_GAP:
      continue
    score = _clip(candidate.risk_score + 0.5 * candidate.on_path_score + 0.25 * candidate.confidence, 0.0, 2.0)
    replacement = LeadReplacementCandidate(
      active=True,
      replacing_idx=replacing.lead_idx,
      candidate_idx=candidate.lead_idx,
      score=score,
      reason="exiting_lead_replacement",
      block_reason="replacement_threat_suppress_only",
    )
    if best is None or replacement.score > best.score:
      best = replacement
  return best if best is not None else LeadReplacementCandidate(replacing_idx=replacing.lead_idx, reason="no_candidate")


def _behavior_rank(state: LeadRelevanceState, lead_dominant_idx: int | None) -> float:
  hint = 0.4 if state.lead_idx == lead_dominant_idx else 0.0
  opening = _clip(state.v_rel / 2.0)
  return 2.0 * state.on_path_score + state.confidence + opening + hint - state.ghost_score


def _alternate_threat(state: LeadRelevanceState, behavior: LeadRelevanceState | None) -> bool:
  if behavior is None:
    return False
  if state.lead_idx == behavior.lead_idx:
    return False
  if _same_lead_duplicate(state, behavior):
    return False
  if not state.suppressive:
    return False
  return bool(state.shadow or state.authority != LEAD_AUTHORITY_PROGRESS_ALLOWED or _is_close_or_closing(state))


def _same_lead_duplicate(state: LeadRelevanceState, behavior: LeadRelevanceState) -> bool:
  if not (state.status and behavior.status):
    return False
  state_track_known = state.track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN
  behavior_track_known = behavior.track_id != LEAD_CONFIDENCE_TRACK_UNKNOWN
  if state_track_known or behavior_track_known:
    return bool(state_track_known and behavior_track_known and state.track_id == behavior.track_id)
  return bool(
    abs(state.d_rel - behavior.d_rel) <= LEAD_CONTEXT_DUPLICATE_D_REL_TOL and
    abs(state.v_lead - behavior.v_lead) <= LEAD_CONTEXT_DUPLICATE_V_TOL and
    abs(state.v_rel - behavior.v_rel) <= LEAD_CONTEXT_DUPLICATE_V_TOL and
    abs(state.path_y_rel - behavior.path_y_rel) <= LEAD_CONTEXT_DUPLICATE_Y_TOL
  )


def _context_reason(physical: LeadRelevanceState | None, behavior: LeadRelevanceState | None, blocked_reason: str) -> str:
  if blocked_reason:
    return blocked_reason
  if behavior is not None:
    return f"behavior_{behavior.reason}"
  if physical is not None:
    return f"physical_{physical.reason}"
  return "no_lead"


def _close_stop_pullaway_progress_allowed(d_rel: float, v_lead: float, v_rel: float,
                                          progress_model: LeadProgressModel) -> bool:
  # Close stopped/crawling leads stay suppressive unless the same stable,
  # on-path progress model already says the lead is safely opening.
  return bool(
    progress_model.allowed and
    progress_model.stop_threat_absent and
    0.0 < d_rel <= LEAD_CONTEXT_CLOSE_STOP_D_REL and
    LEAD_CONTEXT_CLOSE_STOP_PULLAWAY_MIN_V_LEAD < v_lead <= LEAD_CONTEXT_CLOSE_STOP_V and
    v_rel > LEAD_CONTEXT_CLOSE_STOP_PULLAWAY_MIN_OPENING
  )


def _stopped_gap_creep_progress_allowed(d_rel: float, v_lead: float, v_rel: float,
                                        progress_model: LeadProgressModel) -> bool:
  stopped_gap_excess = d_rel - LEAD_CONTEXT_STOP_DISTANCE
  return bool(
    progress_model.confidence_stability_sufficient and
    progress_model.alternate_threat_absent and
    progress_model.shadow_absent and
    0.0 <= v_lead <= LEAD_CONTEXT_CLOSE_STOP_PULLAWAY_MIN_V_LEAD and
    v_rel >= LEAD_CONTEXT_STOP_GAP_CREEP_MIN_V_REL and
    LEAD_CONTEXT_STOP_GAP_CREEP_ARM_EXCESS <= stopped_gap_excess <= LEAD_CONTEXT_STOP_GAP_CREEP_MAX_EXCESS
  )
