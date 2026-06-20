"""Pose-anchored CurveMemory (lateral demand stage).

Corner amnesia is low-speed perception degradation across the whole slow approach->stop->launch
(validated by tools/drive_lab/replay_curve_memory.py), not just the standstill reset — the model's
per-frame curvature collapses below ~1 m/s, before the car even stops. Curvature is geometric
(kappa = dtheta/ds) and speed-independent, so we anchor the road's curvature to cumulative arc length
``s_abs`` (the integral of vEgo): capture ``kappa(s_abs)`` from confident, fast-enough frames where
vision is reliable, then recall ``kappa`` at the *current* s_abs while vision is degraded.

Vision-authority-always: memory only fills the low-confidence window, it only ever *raises* an
under-curved vision toward the remembered corner (never reduces a confident turn), it is vetoed when
confident vision disagrees, and it fails safe to the incoming curvature when pose/memory is
unavailable. Opt-in (``CurveMemoryEnabled``); default-off.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from openpilot.common.realtime import DT_CTRL

BIN_M = 1.0                      # arc-length resolution of the curvature buffer
MAX_AHEAD_M = 22.0               # capture road curvature up to this far ahead (reliable vision range)
MAX_BEHIND_M = 4.0              # keep this much behind ego, prune older
CAPTURE_MIN_V = 4.0             # m/s; only capture from frames fast enough for reliable vision
CAPTURE_MIN_QUALITY = 0.6      # at good speed the curvature is reliable even at moderate soft-quality
#                                (the quality penalty is usually lane-confidence, not curvature error)
CAPTURE_EMA = 0.35             # buffer smoothing per capture
RECALL_MAX_V = 6.0             # m/s; only recall/supplement at/below this speed (the degraded window)
MIN_CURVATURE = 0.008          # only remember/recall real corners (~125 m radius)
MAX_RECALL_CURVATURE = 0.03    # safety clamp on the recalled curvature
VETO_QUALITY = 0.85            # vision at/above this quality can veto a sign-disagreeing memory
STALE_RESET_S = 4.0            # drop the buffer after this long without a valid path (pose drift)
DRIVER_RECALL_INHIBIT_S = 0.4
DEGRADED_RECALL_MIN_WEIGHT = 0.65
HARD_INVALID_RECALL_MIN_WEIGHT = 0.80
CAPTURE_BLOCK_REASONS = frozenset(("invalid_path", "path_disagreement", "frame_drop", "curvature_jump", "high_path_std"))
TRUSTED_CAPTURE_REASONS = frozenset(("ok", "low_lane_confidence"))
SOFT_DEGRADED_RECALL_REASONS = frozenset(("low_lane_confidence", "high_path_std", "frame_drop"))


@dataclass(frozen=True)
class CurveMemoryInputs:
  enabled: bool
  lat_active: bool
  v_ego: float
  desired_curvature: float       # incoming (processed vision) curvature — memory supplements this
  path_quality: float
  path_gated: bool = False
  path_reason: str = "ok"
  steering_pressed: bool | None = None
  lane_change_active: bool = False
  position_x: Sequence[float] = ()
  position_y: Sequence[float] = ()
  orientation_z: Sequence[float] = ()
  valid_path: bool = True


@dataclass(frozen=True)
class CurveMemoryResult:
  desired_curvature: float
  active: bool                   # memory raised the command this frame
  source: str
  remembered: float              # recalled curvature (nan when none)
  s_abs: float
  samples: int                   # buffer occupancy


def _finite(value: object, default: float = 0.0) -> float:
  try:
    v = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError):
    return default
  return v if math.isfinite(v) else default


class CurveMemory:
  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    self._s_abs = 0.0
    self._kappa: dict[int, float] = {}   # bin index -> smoothed road curvature (confident captures)
    self._trusted: dict[int, bool] = {}
    self._invalid_s = 0.0
    self._driver_inhibit_s = 0.0
    self._lane_change_active = False

  def update(self, inputs: CurveMemoryInputs, dt: float = DT_CTRL) -> CurveMemoryResult:
    base = _finite(inputs.desired_curvature)
    if not inputs.enabled:
      self.reset()
      return CurveMemoryResult(base, False, "disabled", math.nan, self._s_abs, 0)

    dt = max(0.0, _finite(dt))
    v_ego = max(0.0, _finite(inputs.v_ego))
    self._s_abs += v_ego * dt

    steering_pressed = inputs.steering_pressed
    driver_unknown = steering_pressed is None
    driver_pressed = bool(steering_pressed)
    if driver_pressed:
      self._driver_inhibit_s = DRIVER_RECALL_INHIBIT_S
    elif self._driver_inhibit_s > 0.0:
      self._driver_inhibit_s = max(0.0, self._driver_inhibit_s - dt)

    if inputs.lane_change_active and not self._lane_change_active:
      self._kappa.clear()
      self._trusted.clear()
      self._invalid_s = 0.0
    self._lane_change_active = bool(inputs.lane_change_active)

    # Drop the buffer if the path has been invalid for a while (pose/geometry no longer trustworthy).
    if inputs.valid_path:
      self._invalid_s = 0.0
    else:
      self._invalid_s += dt
      if self._invalid_s >= STALE_RESET_S:
        self._kappa.clear()
        self._trusted.clear()

    # Capture kappa(s) ahead from confident, fast-enough frames (where vision is reliable).
    if self._capture_allowed(inputs, driver_pressed, driver_unknown):
      self._capture(inputs)

    self._prune()

    recall = self._recall(self._s_abs)
    remembered = recall[0] if recall is not None else None
    remembered_trusted = recall[1] if recall is not None else False
    out, active, source = self._apply(base, remembered, remembered_trusted, inputs, driver_pressed, driver_unknown)
    return CurveMemoryResult(out, active, source, remembered if remembered is not None else math.nan,
                             self._s_abs, len(self._kappa))

  def _capture(self, inputs: CurveMemoryInputs) -> None:
    xs = [_finite(x, math.nan) for x in inputs.position_x]
    ys = [_finite(y, math.nan) for y in inputs.position_y]
    th = [_finite(t, math.nan) for t in inputs.orientation_z]
    n = min(len(xs), len(ys), len(th))
    if n < 3:
      return
    s = 0.0
    for i in range(1, n):
      dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
      if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(th[i]) and math.isfinite(th[i - 1])):
        break
      ds = math.hypot(dx, dy)
      if ds <= 1e-3:
        continue
      s += ds
      if s > MAX_AHEAD_M:
        break
      kappa = (th[i] - th[i - 1]) / ds
      if not math.isfinite(kappa):
        continue
      b = int(round((self._s_abs + s) / BIN_M))
      prev = self._kappa.get(b)
      self._kappa[b] = kappa if prev is None else (prev + CAPTURE_EMA * (kappa - prev))
      self._trusted[b] = inputs.path_reason in TRUSTED_CAPTURE_REASONS

  def _capture_allowed(self, inputs: CurveMemoryInputs, driver_pressed: bool, driver_unknown: bool) -> bool:
    if driver_pressed or driver_unknown or self._driver_inhibit_s > 0.0:
      return False
    if inputs.lane_change_active or not inputs.lat_active:
      return False
    if not inputs.valid_path or inputs.path_gated:
      return False
    if inputs.path_reason in CAPTURE_BLOCK_REASONS:
      return False
    if inputs.path_reason == "low_lane_confidence":
      return inputs.path_quality >= CAPTURE_MIN_QUALITY
    return inputs.path_quality >= CAPTURE_MIN_QUALITY

  def _recall(self, s_abs: float) -> tuple[float, bool] | None:
    if not self._kappa:
      return None
    pos = s_abs / BIN_M
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    klo, khi = self._kappa.get(lo), self._kappa.get(hi)
    if klo is None and khi is None:
      return None
    if klo is None:
      assert khi is not None
      return khi, bool(self._trusted.get(hi, False))
    if khi is None or lo == hi:
      return klo, bool(self._trusted.get(lo, False))
    frac = pos - lo
    trusted = bool(self._trusted.get(lo, False) and self._trusted.get(hi, False))
    return klo + frac * (khi - klo), trusted

  def _apply(self, base: float, remembered: float | None, remembered_trusted: bool, inputs: CurveMemoryInputs,
             driver_pressed: bool, driver_unknown: bool) -> tuple[float, bool, str]:
    v_ego = max(0.0, _finite(inputs.v_ego))
    if driver_pressed or driver_unknown:
      return base, False, "driver_override"
    if self._driver_inhibit_s > 0.0:
      return base, False, "driver_recall_inhibit"
    if inputs.lane_change_active:
      return base, False, "lane_change"
    if remembered is None or abs(remembered) < MIN_CURVATURE or v_ego > RECALL_MAX_V:
      return base, False, "vision"
    remembered = math.copysign(min(abs(remembered), MAX_RECALL_CURVATURE), remembered)

    # Veto: confident vision that disagrees in sign owns the frame (the corner ended / reversed).
    if inputs.path_quality >= VETO_QUALITY and abs(base) >= MIN_CURVATURE and base * remembered < 0.0:
      return base, False, "vetoed"

    # Memory only RAISES an under-curved vision toward the remembered corner (never reduces a turn).
    same_dir_target = base * remembered >= 0.0 or abs(base) < MIN_CURVATURE
    if not (same_dir_target and abs(remembered) > abs(base)):
      return base, False, "vision"

    # Fill weight: full when vision is degraded, fading out as quality climbs toward the veto bar;
    # clamped between vision and the remembered corner. Downstream clip_curvature rate-limits.
    weight = 1.0 - max(0.0, min(1.0, inputs.path_quality / VETO_QUALITY))
    if remembered_trusted and inputs.path_reason in SOFT_DEGRADED_RECALL_REASONS:
      weight = max(weight, DEGRADED_RECALL_MIN_WEIGHT)
    elif remembered_trusted and inputs.path_reason == "invalid_path" and self._invalid_s < STALE_RESET_S:
      weight = max(weight, HARD_INVALID_RECALL_MIN_WEIGHT)
    out = math.copysign(min(abs(base + weight * (remembered - base)), abs(remembered)), remembered)
    if abs(out) <= abs(base) + 1e-9:
      return base, False, "vision"
    return out, True, "memory"

  def _prune(self) -> None:
    if not self._kappa:
      return
    lo = int(math.floor((self._s_abs - MAX_BEHIND_M) / BIN_M))
    hi = int(math.ceil((self._s_abs + MAX_AHEAD_M + BIN_M) / BIN_M))
    for b in [b for b in self._kappa if b < lo or b > hi]:
      del self._kappa[b]
      self._trusted.pop(b, None)
