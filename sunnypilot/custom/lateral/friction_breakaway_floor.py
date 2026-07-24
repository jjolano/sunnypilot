"""Friction breakaway floor — anti-stick-slip shaping for small persistent errors.

Route 000002a1 (2026-07-17) showed rack stick-slip: the legacy low-demand friction
scaling (``response_core.low_demand_friction_scale``, quadratic suppression) starves
small corrections to ~11% of breakaway torque, so the wheel waits for P/I to wind
through the sticky zone and then jumps (0.6-1.2 deg steps, amplitude-dependent lag).

The legacy scaling exists to prevent friction sign-chatter at noise-level errors.
This floor keeps that protection but restores breakaway for *persistent* small
errors: only when the error sign has held for ``PERSIST_FRAMES`` above
``MIN_ERROR`` does the friction term get floored at ``FLOOR_FRAC`` of full
breakaway. Sign flips and near-zero errors reset the persistence counter, so
noise-level chatter never sees the floor.

Modes (``LatFrictionBreakawayMode`` param, fail-closed to off):
  off    — legacy behavior, floor never computed into the command.
  shadow — floor computed and logged (frictionFloorActive/Delta), never applied.
  apply  — floored friction term feeds the feedforward.
"""
from __future__ import annotations

from dataclasses import dataclass

# Tuned in tools/drive_lab/stiction_lab.py against the route 000002a1 signature.
# A hard floor dithers (post-breakout error flip re-engages the floor the other
# way); instead the boost is a continuous steep ramp in |error|, slew-limited.
#
# 0.7 was a margin choice: 0.9 was stable in sim, but the sim's breakaway
# conveniently matched the learned friction, so the real rack was unknown. The
# observer has since measured it (routes 2cd+2ce, 98 events): left 0.286 / right
# 0.163 normalized, i.e. the real rack is *stiffer* than the 0.13 the lab assumed,
# so the margin was being spent in the wrong direction. Re-swept the lab at both
# measured breakaways across wander/curve x amplitude x road crown: 0.9 lowers
# desired-vs-actual lag in every cell (e.g. 0.30 -> 0.18 s at amp 0.2 / breakaway
# 0.286) with dwell-jump rate, rate_hf/lf and tracking RMSE flat or better.
# Safe only together with DIRECTION_FRAC_MAX_TOTAL below - uncapped, the learned
# left/right skew pushed the strong direction to 1.15x full breakaway and the
# high-demand cell regressed (rate_hf/lf 1.06 -> 1.27).
FLOOR_FRAC = 0.9        # max boost target, fraction of full breakaway
# MIN_ERROR is the noise deadband, but PERSIST_FRAMES below is the real noise gate:
# nothing is boosted until the error sign has held for 150 ms. Routes 2cd+2ce show
# 0.03 was locking 38.5% of engaged driving out of the floor, and the band it
# excluded is signal, not noise — errors in [0.015, 0.03) hold their sign for
# >=150 ms 76.5% of the time, close to the already-admitted [0.03, 0.06) band
# (91.8%) and far above the flickering [0.005, 0.015) band (57.7%). The 23.5% that
# do flicker are still rejected by persistence, so this only removes a redundant
# second gate. Lab (both measured breakaways, with and without road crown): lag
# 0.24 -> 0.15 s at amp 0.12, dwell-jump rate and rate_hf/lf flat, RMSE slightly
# better, steady bias unchanged (0.0038 -> 0.0040 at crown 0.06). 0.01 was a step
# too far: it regressed lag 0.15 -> 0.22 s at the stiff breakaway under crown.
MIN_ERROR = 0.015       # m/s^2 — below this is noise, never boosted
ERROR_RAMP = 0.12       # m/s^2 of |error| above MIN_ERROR for full boost
PERSIST_FRAMES = 15     # 150 ms of sustained error sign at 100 Hz
SLEW_PER_FRAME = 0.015  # lat-accel units per frame (~1.5/s at 100 Hz)

# Direction-aware scaling from the breakaway observer profile. Route 000002ac:
# left breakaway median 0.299 vs right 0.206 normalized — the rack needs ~1.45x
# more torque to break out leftward. Per-direction fracs scale FLOOR_FRAC by each
# median over their mean, clamped so a skewed profile can't starve or saturate a
# direction.
DIRECTION_FRAC_MIN_SCALE = 0.5
DIRECTION_FRAC_MAX_SCALE = 1.3
# Absolute ceiling on the scaled per-direction frac. The floor exists to reach
# breakaway, never to exceed it: boosting past the full learned breakaway torque
# buys no extra unsticking and starts to hunt. The measured left/right skew is
# already 1.75x (scale 1.274), so without this ceiling FLOOR_FRAC 0.9 would ask
# for 1.15x full breakaway leftward.
DIRECTION_FRAC_MAX_TOTAL = 1.0
# Sign mapping, derived from the response-core conventions (see torque_v2_1
# measured_curvature = -calc_curvature(...)): positive error = rightward
# correction = wheel about to move right = the observer's *right* median;
# negative error = leftward = observer *left* median. The observer keys motion
# direction by sign(steeringRateDeg), where positive rate = wheel moving left.


@dataclass(frozen=True)
class FrictionFloorDebug:
  active: bool
  delta: float  # lat-accel units the floor adds (would add, in shadow)


class FrictionBreakawayFloor:
  def __init__(self, floor_frac: float = FLOOR_FRAC, min_error: float = MIN_ERROR,
               error_ramp: float = ERROR_RAMP, persist_frames: int = PERSIST_FRAMES,
               slew_per_frame: float = SLEW_PER_FRAME):
    self.floor_frac = floor_frac
    self.min_error = min_error
    self.error_ramp = error_ramp
    self.persist_frames = persist_frames
    self.slew_per_frame = slew_per_frame
    self.mode = "off"
    self._count = 0
    self._sign = 0
    self._boost = 0.0
    # per-error-sign frac; symmetric until a valid breakaway profile arrives
    self._frac = {1: self.floor_frac, -1: self.floor_frac}
    self.debug = FrictionFloorDebug(False, 0.0)

  def reset(self) -> None:
    self._count = 0
    self._sign = 0
    self._boost = 0.0
    self.debug = FrictionFloorDebug(False, 0.0)

  def apply_profile(self, profile) -> None:
    """Scale per-direction fracs from a parsed breakaway profile; None = symmetric."""
    if not profile:
      self._frac = {1: self.floor_frac, -1: self.floor_frac}
      return
    left, right = profile['left'], profile['right']
    mean = (left + right) / 2.0
    if mean <= 0:
      self._frac = {1: self.floor_frac, -1: self.floor_frac}
      return

    def scaled(median: float) -> float:
      scale = min(max(median / mean, DIRECTION_FRAC_MIN_SCALE), DIRECTION_FRAC_MAX_SCALE)
      return min(self.floor_frac * scale, DIRECTION_FRAC_MAX_TOTAL)

    # positive error = rightward correction -> right median; negative -> left
    self._frac = {1: scaled(right), -1: scaled(left)}

  def shape(self, base_term: float, error: float, torque_params) -> float:
    """Return the friction term to use; records shadow/apply debug either way."""
    if self.mode == "off":
      self.reset()
      return base_term

    sign = 1 if error > self.min_error else (-1 if error < -self.min_error else 0)
    if sign == 0 or sign != self._sign:
      self._sign = sign
      self._count = 1 if sign != 0 else 0
    else:
      self._count += 1

    target = 0.0
    if self._sign != 0 and self._count >= self.persist_frames:
      full = abs(torque_params.friction * torque_params.latAccelFactor)
      ramp = min(1.0, (abs(error) - self.min_error) / self.error_ramp)
      target = self._sign * self._frac[self._sign] * full * max(ramp, 0.0)

    # slew toward target so engagement/release never injects a torque step
    delta = target - self._boost
    self._boost += max(-self.slew_per_frame, min(self.slew_per_frame, delta))

    # the boost only ever deepens the friction term in the error direction
    if self._boost > 0:
      floored = max(base_term, self._boost)
    elif self._boost < 0:
      floored = min(base_term, self._boost)
    else:
      floored = base_term

    self.debug = FrictionFloorDebug(abs(floored - base_term) > 1e-9, floored - base_term)
    return floored if self.mode == "apply" else base_term
