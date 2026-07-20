"""Longitudinal model-evidence trust gate (Phase 5).

The longitudinal analog of the lateral ``model_path_processor`` quality gating: the model's
stop / slowdown is treated as ONE evidence source, not ground truth. Trust is earned
per-signal and applied asymmetrically — caution is honored cheaply, relaxation expensively —
so a low-confidence or contradicted model prediction degrades gracefully toward a
conservative precautionary decel instead of being obeyed or ignored outright. This directly
targets the "over-trust the model in low-quality regimes (occlusion / flicker / glare)"
failure that likely accounts for some unattributed quirks.

Principles (from the trajectory/trust review):
- Cross-validate against non-learned signals (radar d_rel/v_rel, kinematics).
- Disagreement => lower trust, fall back toward the conservative envelope.
- Trust caution cheaply; require corroboration to honor a full hard stop.
- Blend by confidence (no hard model-on/off switching -> no discontinuity/oscillation).
- Relaxation can never override the reactive/MPC lead-follow floor (that binds in the
  decision core independently), so this gate only shapes the *caution* side.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

from openpilot.sunnypilot.custom.longitudinal.lead_context import _path_relative_y_or_none

GENTLE_CAUTION_DECEL = -0.4     # precautionary decel for a low-confidence model slowdown
TRUST_FULL_STOP = 0.7           # stop_prob/trust above which a hard should_stop is honored
RADAR_CORROBORATION_TRUST = 0.85
LEAD_CLOSING_MIN = 0.25         # m/s relative closing to count as radar corroboration

# CautionRamp: rate-limit how fast the caution floor may deepen below GENTLE_CAUTION_DECEL.
CAUTION_RAMP_DEEPEN_RATE = 0.45   # m/s^2 per s of sustained model slowdown demand
CAUTION_RAMP_RELEASE_RATE = 2.0   # m/s^2 per s back toward gentle once the demand lifts
CAUTION_RAMP_FLOOR_MIN = -2.5

# CorroborationHold: how long a closing radar echo keeps earned caution depth unlocked.
# Covers the longest observed radar-flicker gap on a real stopped queue (1.6 s) with margin.
CORROBORATION_HOLD_S = 2.5

# CutOutCautionRecovery: unwind earned caution fast when the corroborating lead cuts out.
# Route 296 t=848: a braking turner (path-relative |y| to 1.8 m) left the lane; radar
# dropped it and the road was clear, but the model's desired accel lagged ~2.4 s and the
# earned caution floor kept commanding -1.1 until the driver gas-overrode. A genuine
# lateral exit is the opposite of the radar flicker CorroborationHold protects against
# (flickers keep |path y| small), so it cancels earned depth instead of extending it.
CUT_OUT_RECOVERY_S = 2.5           # s; bounded window of gentle-capped caution after a cut-out
CUT_OUT_EXIT_MIN_PATH_Y = 1.2      # m; lateral path offset marking a genuine exit, not flicker
CUT_OUT_EXIT_LOOKBACK_S = 1.0      # s; window before disappearance searched for exit evidence
CUT_OUT_MIN_CLOSING = 0.5          # m/s; the departed lead must have been a closing context
CUT_OUT_MIN_D_REL = 15.0           # m; near leads vanish under the radar nose at stops — never those

# StopTrustLearner: learn how much to trust the upstream model stop from driver disagreement.
STOP_TRUST_INITIAL = 0.8
STOP_TRUST_MIN = 0.2
STOP_TRUST_MAX = 0.95
STOP_TRUST_DISAGREE_RATE = 0.5   # confidence/s drop when the driver countermands a model stop
STOP_TRUST_AGREE_RATE = 0.05     # confidence/s recovery when the driver accepts it


@dataclass(frozen=True)
class ModelStopTrustResult:
  should_stop: bool
  desired_accel: float   # trust-scaled: gentle at low trust, full model decel at high trust
  trust: float
  reason: str


def _clip(v: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, v))


def gate_model_stop(model_should_stop: bool, model_desired_accel: float, stop_prob: float,
                    has_radar_lead: bool = False, lead_v_rel: float = 0.0,
                    model_stale: bool = False) -> ModelStopTrustResult:
  """Trust-gate the model's stop/slowdown.

  ``stop_prob`` is the model's confidence in the stop (modelV2). ``lead_v_rel`` < 0 means a
  radar lead closing (physical corroboration of a slowdown)."""
  stop_prob = _clip(float(stop_prob), 0.0, 1.0)
  model_decel = min(0.0, float(model_desired_accel))

  # Model says go (no slowdown): nothing to gate here. Any real hazard (radar lead, map,
  # curve) binds independently in the decision core, so we never relax safety here.
  if not model_should_stop and model_decel >= 0.0:
    return ModelStopTrustResult(False, float(model_desired_accel), 1.0, "model_clear")

  # A stale model stop is semantic evidence that arrived too late. Do not commit a stop or
  # honor hard model decel from it; radar/MPC lead physics remains a separate safety floor.
  if model_stale:
    return ModelStopTrustResult(False, GENTLE_CAUTION_DECEL, 0.0, "model_stale")

  # Caution: the model wants to slow/stop. Earn trust from model confidence, raised by radar
  # corroboration (a closing radar lead physically agrees with slowing down).
  trust = stop_prob
  reason = "model_only"
  if has_radar_lead and float(lead_v_rel) < -LEAD_CLOSING_MIN:
    trust = max(trust, RADAR_CORROBORATION_TRUST)
    reason = "radar_corroborated"

  # Honor a full hard stop only when trusted; otherwise hold the stop flag back (still decel,
  # just not a committed stop) to avoid slamming on a flickery model stop.
  should_stop = bool(model_should_stop and trust >= TRUST_FULL_STOP)

  # Blend the commanded decel from a gentle precautionary value (low trust) to the full model
  # decel (high trust). Never command less caution than the model asks if it is fully trusted.
  desired_accel = GENTLE_CAUTION_DECEL + trust * (model_decel - GENTLE_CAUTION_DECEL)
  return ModelStopTrustResult(should_stop, float(desired_accel), float(trust), reason)


class CautionRamp:
  """Rate-limited caution floor for sustained model slowdown demand.

  Route 261: leadless stop approaches pinned the model-stop candidate at
  GENTLE_CAUTION_DECEL (-0.4) while the model's demand ramped to -2.0 (no in-horizon
  rest point => no stop distance), then banged to the -1.5 stop floor on single frames
  when a rest point flickered in. This ramp lets the caution floor *earn* depth: it
  deepens toward the model's demand only while the demand persists, and releases fast
  the moment the model lifts. Flickers deepen it by ~0.1 m/s^2 and are then released;
  a real 5-6 s urban stop approach tracks the model demand smoothly. Hard trusted stop
  commits bypass the floor entirely."""

  def __init__(self):
    self.floor = GENTLE_CAUTION_DECEL

  def update(self, model_desired_accel: float, dt: float) -> float:
    target = _clip(float(model_desired_accel), CAUTION_RAMP_FLOOR_MIN, GENTLE_CAUTION_DECEL)
    dt = max(0.0, float(dt))
    if target < self.floor:
      self.floor = max(target, self.floor - CAUTION_RAMP_DEEPEN_RATE * dt)
    else:
      self.floor = min(target, self.floor + CAUTION_RAMP_RELEASE_RATE * dt)
    return self.floor


class CorroborationHold:
  """Radar-corroboration latch for CautionRamp-earned stop depth.

  Earned depth past the -1.5 uncommitted stop floor is only trustworthy when something
  non-learned agrees a stop is real. A closing radar echo is that signal, but real stopped
  queues hold radar lock only intermittently (route 28c: closing echoes on ~15% of frames,
  gaps up to 1.6 s), so a per-frame gate would re-pin the floor mid-approach and oscillate.
  Latch instead: any closing echo unlocks depth for CORROBORATION_HOLD_S. A vision-only
  hallucination never gets an echo and stays capped at the stop floor."""

  def __init__(self):
    self.hold_s = 0.0

  def update(self, closing_radar_lead: bool, dt: float) -> bool:
    if closing_radar_lead:
      self.hold_s = CORROBORATION_HOLD_S
    else:
      self.hold_s = max(0.0, self.hold_s - max(0.0, float(dt)))
    return self.hold_s > 0.0


class CutOutCautionRecovery:
  """Bounded gentle-cap window on uncommitted model caution after the lead cuts out.

  Watches leadOne; when a closing lead departs with lateral-exit evidence (path-relative
  |y| beyond ``CUT_OUT_EXIT_MIN_PATH_Y`` within the last second — a genuine cut-out, not a
  flicker), returns True for ``CUT_OUT_RECOVERY_S`` so the caller caps uncommitted caution
  at ``GENTLE_CAUTION_DECEL`` while the model catches up to the cleared scene. Trusted stop
  commits bypass the caution floor entirely and are unaffected. Any closing lead present
  cancels the window immediately. Fail-closed: any fault returns False (no cap)."""

  def __init__(self):
    self._t = 0.0
    self._recent: deque[tuple[float, float, bool, float]] = deque(maxlen=32)
    self._had_lead = False
    self._recovery_s = 0.0

  def update(self, lead_one: Any, model_msg: Any, dt: float) -> bool:
    try:
      step = max(0.0, float(dt)) if math.isfinite(float(dt)) else 0.0
      self._t += step
      self._recovery_s = max(0.0, self._recovery_s - step)
      if lead_one is not None and bool(getattr(lead_one, "status", False)):
        d_rel = float(getattr(lead_one, "dRel", 0.0) or 0.0)
        y_rel = float(getattr(lead_one, "yRel", 0.0) or 0.0)
        v_rel = float(getattr(lead_one, "vRel", 0.0) or 0.0)
        if not all(math.isfinite(v) for v in (d_rel, y_rel, v_rel)):
          raise ValueError("non-finite lead")
        path_y = _path_relative_y_or_none(y_rel, d_rel, model_msg)
        abs_y = abs(path_y) if path_y is not None else abs(y_rel)
        closing = -v_rel > CUT_OUT_MIN_CLOSING
        self._recent.append((self._t, abs_y, closing, d_rel))
        if closing:
          self._recovery_s = 0.0  # a closing lead is live evidence; never cap caution for it
        self._had_lead = True
      else:
        if self._had_lead:
          recent = [s for s in self._recent if self._t - s[0] <= CUT_OUT_EXIT_LOOKBACK_S]
          was_closing = any(c for _, _, c, _ in recent)
          exit_y = max((y for _, y, _, _ in recent), default=0.0)
          far = max((d for _, _, _, d in recent), default=0.0) >= CUT_OUT_MIN_D_REL
          if was_closing and far and exit_y >= CUT_OUT_EXIT_MIN_PATH_Y:
            self._recovery_s = CUT_OUT_RECOVERY_S
        self._had_lead = False
        self._recent.clear()
      return self._recovery_s > 0.0
    except Exception:
      self._recent.clear()
      self._had_lead = False
      self._recovery_s = 0.0
      return False


class StopTrustLearner:
  """Learn the confidence to feed gate_model_stop from real driver disagreement.

  Rather than guessing a model stop probability, we take upstream's verified model stop
  (modelV2.action.shouldStop) at face value and adjust how much to trust it from how the
  driver reacts: a driver who countermands a model stop (gas, or disengages) during the stop
  is telling us the stop was wrong -> drop confidence fast; a driver who lets it happen agrees
  -> recover slowly. Over a drive this softens repeatedly-false model stops while keeping the
  ones the driver accepts. (Session-scoped; param persistence is a later enhancement.)"""

  def __init__(self, initial: float = STOP_TRUST_INITIAL):
    self.confidence = _clip(float(initial), STOP_TRUST_MIN, STOP_TRUST_MAX)

  def update(self, model_should_stop: bool, driver_disagrees: bool, dt: float) -> float:
    if model_should_stop:
      rate = -STOP_TRUST_DISAGREE_RATE if driver_disagrees else STOP_TRUST_AGREE_RATE
      self.confidence = _clip(self.confidence + rate * max(0.0, float(dt)), STOP_TRUST_MIN, STOP_TRUST_MAX)
    return self.confidence


# Model-stop anchor: routes 2b5/2b2/2ac/2a9/2aa/2b0/296 (8 engaged leadless model stops)
# all show the same signature — the model's predicted rest point starts optimistic and
# firms nearer through the approach, so demand tracking it verbatim under-brakes by
# 0.4-0.7 m/s^2 for the first 4-8 s and repays at -1.5..-2.0 late. The anchor converts
# that optimism into early, visible decel and commits to the stop point once seen.
STOP_ANCHOR_CONSERVATIVE_FRACTION = 0.85  # plan to this fraction of the model's distance...
STOP_ANCHOR_MAX_SHRINK_M = 12.0           # ...but never more than this much nearer
STOP_ANCHOR_JUMP_M = 15.0                 # a jump past this (either direction) needs...
STOP_ANCHOR_JUMP_CONFIRM_FRAMES = 3       # ...this many consecutive frames (jitter guard)
STOP_ANCHOR_MAX_DIVERGENCE_M = 15.0       # anchor never commits further below the live target...
STOP_ANCHOR_DIVERGENCE_FRACTION = 0.5     # ...nor below this fraction of it (binds near the stop)
STOP_ANCHOR_RELEASE_MISSING_S = 1.0       # sustained model retraction (green) releases
STOP_ANCHOR_MIN_COMMIT_S = 0.25           # consumers ignore commitments younger than this (blip filter)
# Route 2ba t=1517/1623: the committed distance burned to 0 with travel while the model still
# placed the stop 6.6 m ahead, releasing the whole stop posture at 4.5 m/s (driver braked).
# A committed stop may never report "arrived" while ego is still moving.
STOP_ANCHOR_MIN_ACTIVE_M = 2.0
STOP_ANCHOR_MIN_ACTIVE_V_EGO = 0.5
# Travel-consistency corroboration: a real stop line's predicted distance shrinks ~1:1 with
# ego travel; a hallucination's does not (the phantom signature is exactly that failure).
STOP_ANCHOR_CORR_MIN_TRAVEL_M = 8.0       # earn after this much travel under commitment...
STOP_ANCHOR_CORR_MIN_SHRINK_RATIO = 0.6   # ...with the raw point shrinking at least this share of it


class ModelStopAnchor:
  """Commit-and-ratchet for the model's predicted stop point (red lights / stop signs).

  ``update`` returns the anchored, conservative stop distance the policy should plan to,
  or None when no stop is committed. The anchored point advances with ego travel every
  frame and holds its commitment against small shallowing drift of the model's point; it
  may never diverge more than ``STOP_ANCHOR_MAX_DIVERGENCE_M`` below the live
  conservative target, so a genuinely receding stop point (the phantom shape: reported
  distance not shrinking with travel) is followed with bounded frontload instead of being
  inverted into an in-rushing one. Jumps beyond ``STOP_ANCHOR_JUMP_M`` in either
  direction need ``STOP_ANCHOR_JUMP_CONFIRM_FRAMES`` of corroboration so one bad frame
  can neither slam nor release the demand, and a sustained model retraction
  (``STOP_ANCHOR_RELEASE_MISSING_S``, the green light) releases entirely so the
  departing-lead reaccel behaviors stay responsive.

  ``corroborated`` latches once the raw stop point has shrunk with ego travel
  (``STOP_ANCHOR_CORR_*``): physical evidence the point is world-fixed, used by wiring to
  unlock earned caution depth past the vision-only floor. A confirmed jump rebases it."""

  def __init__(self):
    self.remaining: float | None = None
    self.committed_s = 0.0
    self.corroborated = False
    self._missing_s = 0.0
    self._jump_frames = 0
    self._corr_d0: float | None = None
    self._corr_travel = 0.0

  def reset(self) -> None:
    self.remaining = None
    self.committed_s = 0.0
    self.corroborated = False
    self._missing_s = 0.0
    self._jump_frames = 0
    self._corr_d0 = None
    self._corr_travel = 0.0

  def _floored(self, v_ego: float) -> float:
    # A committed stop never reports "arrived" while still moving: burn-down past this floor
    # would drop the whole stop posture mid-stop (route 2ba: release at 4.5 m/s, 6.6 m short).
    if self.remaining is not None and float(v_ego) > STOP_ANCHOR_MIN_ACTIVE_V_EGO:
      self.remaining = max(self.remaining, STOP_ANCHOR_MIN_ACTIVE_M)
    return self.remaining

  def update(self, model_stop_distance: float | None, v_ego: float, dt: float) -> float | None:
    dt = max(0.0, float(dt))
    travel = max(0.0, float(v_ego)) * dt
    if self.remaining is not None:
      self.committed_s += dt
      self._corr_travel += travel
    d = float(model_stop_distance) if model_stop_distance is not None else math.nan
    if not math.isfinite(d) or d <= 0.0:
      if self.remaining is None:
        return None
      self._missing_s += dt
      if self._missing_s >= STOP_ANCHOR_RELEASE_MISSING_S:
        self.reset()
        return None
      # brief dropout: hold the commitment, advancing with travel
      self.remaining = max(self.remaining - travel, 0.0)
      return self._floored(v_ego)
    self._missing_s = 0.0
    target = max(d * STOP_ANCHOR_CONSERVATIVE_FRACTION, d - STOP_ANCHOR_MAX_SHRINK_M)
    if self.remaining is None:
      self.remaining = max(target, 0.0)
      self._jump_frames = 0
      self._corr_d0 = d
      self._corr_travel = 0.0
      return self._floored(v_ego)
    if (not self.corroborated and self._corr_d0 is not None
        and self._corr_travel >= STOP_ANCHOR_CORR_MIN_TRAVEL_M
        and (self._corr_d0 - d) >= STOP_ANCHOR_CORR_MIN_SHRINK_RATIO * self._corr_travel):
      self.corroborated = True
    advanced = max(self.remaining - travel, 0.0)
    if abs(target - advanced) > STOP_ANCHOR_JUMP_M:
      self._jump_frames += 1
      if self._jump_frames < STOP_ANCHOR_JUMP_CONFIRM_FRAMES:
        self.remaining = advanced  # unconfirmed jump: hold the commitment
        return self._floored(v_ego)
      # confirmed jump: a genuinely different stop point; consistency re-earns from here
      self.corroborated = False
      self._corr_d0 = d
      self._corr_travel = 0.0
    else:
      self._jump_frames = 0
    if target <= advanced:
      self.remaining = max(target, 0.0)
    else:
      # binds at 15 m far out, proportionally tighter near the stop so the commitment can
      # never burn to zero against a live target a few meters ahead
      divergence = min(STOP_ANCHOR_MAX_DIVERGENCE_M, STOP_ANCHOR_DIVERGENCE_FRACTION * target)
      self.remaining = max(advanced, target - divergence)
    return self._floored(v_ego)
