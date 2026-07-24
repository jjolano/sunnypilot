from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence

import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.custom.lateral.demand.types import (
  DEMAND_SOURCE_LANE_FIT,
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
)


LATERAL_PREVIEW_ASSIST_MODES = frozenset(("off", "shadow", "apply"))
LATERAL_PREVIEW_ASSIST_MIN_SPEED = 5.0
LATERAL_PREVIEW_ASSIST_MIN_PATH_QUALITY = 0.85
LATERAL_PREVIEW_ASSIST_MAX_MODEL_AGE_S = 0.20
# Route 274 replay: 0.08 worsened p95 tracking error and pushed a hard-reset nudge
# above 8 m/s^3. Keep the proven lower authority until an engaged A/B supports more.
LATERAL_PREVIEW_ASSIST_MAX_DELTA_AY = 0.05
LATERAL_PREVIEW_ASSIST_SLEW_LAT_JERK = 0.30
LATERAL_PREVIEW_ASSIST_SIGN_CONFLICT_AY = 0.05
# steer_limited flaps frame-to-frame on the EPS (route 274: ~19 applied-flag toggles per
# second, the top lateral-jerk source). Require it sustained before it blocks the preview.
LATERAL_PREVIEW_ASSIST_STEER_LIMITED_BLOCK_S = 0.20
LATERAL_PREVIEW_ASSIST_COMFORT_HORIZON_BP = [5.0, 15.0, 30.0]
LATERAL_PREVIEW_ASSIST_COMFORT_HORIZON_S = [0.20, 0.35, 0.55]
LATERAL_PREVIEW_ASSIST_GAIN_BP = [5.0, 15.0, 30.0]
LATERAL_PREVIEW_ASSIST_GAIN = [0.75, 0.90, 1.00]
LATERAL_PREVIEW_ASSIST_T_PREVIEW_MIN_S = 0.25
LATERAL_PREVIEW_ASSIST_T_PREVIEW_MAX_S = 1.20


def sanitize_lateral_preview_assist_mode(mode: object) -> str:
  try:
    mode_s = str(mode).strip().lower()
  except Exception:
    return "off"
  return mode_s if mode_s in LATERAL_PREVIEW_ASSIST_MODES else "off"


def _finite_float(value: Any) -> float | None:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return None
  return value if math.isfinite(value) else None


def _finite_array(values: Sequence[Any]) -> list[float] | None:
  result: list[float] = []
  for value in values:
    f = _finite_float(value)
    if f is None:
      return None
    result.append(f)
  return result


def _preview_result(mode: str, reason: str) -> PreviewAssistResult:
  return PreviewAssistResult(mode, False, False, reason, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


def inactive_preview_assist_result(mode: str = "off", reason: str = "disabled") -> PreviewAssistResult:
  return _preview_result(mode, reason)


@dataclass(frozen=True)
class PreviewAssistResult:
  mode: str
  active: bool
  applied: bool
  reason: str
  confidence: float
  t_preview: float
  base_curvature: float
  preview_curvature: float
  curvature_nudge: float
  ay_base: float
  ay_preview: float
  ay_delta: float
  slew_limited: bool

  def debug_dict(self) -> dict[str, float | str | bool]:
    return {
      "lateral_preview_assist_mode": self.mode,
      "lateral_preview_assist_active": self.active,
      "lateral_preview_assist_applied": self.applied,
      "lateral_preview_assist_reason": self.reason,
      "lateral_preview_assist_confidence": self.confidence,
      "lateral_preview_assist_t_preview": self.t_preview,
      "lateral_preview_assist_base_curvature": self.base_curvature,
      "lateral_preview_assist_preview_curvature": self.preview_curvature,
      "lateral_preview_assist_curvature_nudge": self.curvature_nudge,
      "lateral_preview_assist_ay_base": self.ay_base,
      "lateral_preview_assist_ay_preview": self.ay_preview,
      "lateral_preview_assist_ay_delta": self.ay_delta,
      "lateral_preview_assist_slew_limited": self.slew_limited,
    }


class PreviewAssistTracker:
  def __init__(self, dt: float) -> None:
    self.dt = max(float(dt), 1e-3)
    self._steer_limited_block_frames = max(1, int(round(LATERAL_PREVIEW_ASSIST_STEER_LIMITED_BLOCK_S / self.dt)))
    self.reset()

  def reset(self) -> None:
    self._last_output_ay_delta: float | None = None
    self._last_mode = "off"
    self._steer_limited_frames = 0

  def _soft_release(self, mode: str, reason: str, baseline_curvature: float, v_ego: float) -> PreviewAssistResult:
    """Transient blocker: decay the nudge to zero at the entry slew rate instead of
    dropping it in one frame (route 274: instant removal was the top lateral-jerk source).
    Keeps emitting the decaying nudge in apply mode; state clears itself on convergence."""
    output_ay_delta, slew_limited = self._slew_output(0.0)
    speed_sq = max(v_ego * v_ego, 1.0)
    curvature_nudge = output_ay_delta / speed_sq
    preview_curvature = baseline_curvature + curvature_nudge
    ay_base = baseline_curvature * speed_sq
    ay_preview = preview_curvature * speed_sq
    applied = mode == "apply" and abs(curvature_nudge) > 0.0
    reason_out = f"releasing_{reason}" if applied else reason
    return PreviewAssistResult(
      mode=mode, active=False, applied=applied, reason=reason_out,
      confidence=0.0, t_preview=0.0, base_curvature=baseline_curvature,
      preview_curvature=preview_curvature, curvature_nudge=curvature_nudge,
      ay_base=ay_base, ay_preview=ay_preview, ay_delta=output_ay_delta, slew_limited=slew_limited,
    )

  def _slew_output(self, target_ay_delta: float) -> tuple[float, bool]:
    prev = self._last_output_ay_delta
    if prev is None or not math.isfinite(prev):
      prev = 0.0
    max_step = LATERAL_PREVIEW_ASSIST_SLEW_LAT_JERK * self.dt
    output_ay_delta = prev + float(np.clip(target_ay_delta - prev, -max_step, max_step))
    slew_limited = abs(output_ay_delta - target_ay_delta) > 1e-9
    if slew_limited or abs(target_ay_delta) > 0.0:
      self._last_output_ay_delta = output_ay_delta
    else:
      self._last_output_ay_delta = None
    return output_ay_delta, slew_limited

  def _preview_curvature(self, inputs: Any, t_preview: float, v_ego: float) -> float | None:
    yaws = _finite_array(getattr(inputs, "orientation_z", ()) or ())
    yaw_rates = _finite_array(getattr(inputs, "orientation_rate_z", ()) or ())
    if yaws is None or yaw_rates is None:
      return None
    if len(yaws) != len(ModelConstants.T_IDXS) or len(yaw_rates) != len(ModelConstants.T_IDXS):
      return None
    try:
      curvature = float(get_curvature_from_plan(yaws, yaw_rates, ModelConstants.T_IDXS, v_ego, t_preview))
    except Exception:
      return None
    return curvature if math.isfinite(curvature) else None

  def update(self, inputs: Any, model_path_result: Any, demand_source: str, baseline_curvature: float) -> PreviewAssistResult:
    mode = sanitize_lateral_preview_assist_mode(getattr(inputs, "lateral_preview_assist_mode", "off"))
    if mode != self._last_mode:
      self._last_output_ay_delta = None
      self._last_mode = mode
    baseline_curvature_f = _finite_float(baseline_curvature)
    v_ego = _finite_float(getattr(inputs, "v_ego", None))
    if mode == "off":
      self.reset()
      return _preview_result(mode, "disabled")
    if baseline_curvature_f is None or v_ego is None:
      self.reset()
      return _preview_result(mode, "invalid")

    if demand_source == DEMAND_SOURCE_LANE_FIT:
      self.reset()
      return _preview_result(mode, "lane_fit_source")
    if demand_source == DEMAND_SOURCE_LATERAL_MANEUVER:
      self.reset()
      return _preview_result(mode, "maneuver_override")
    if demand_source != DEMAND_SOURCE_MODEL_PATH:
      self.reset()
      return _preview_result(mode, str(demand_source))

    if not bool(getattr(inputs, "lat_active", False)):
      self.reset()
      return _preview_result(mode, "inactive")
    if v_ego < LATERAL_PREVIEW_ASSIST_MIN_SPEED:
      self.reset()
      return _preview_result(mode, "low_speed")

    # Hard stops: driver intent / lane change — drop the nudge immediately and clear state.
    if not bool(getattr(inputs, "lane_change_state_valid", False)):
      self.reset()
      return _preview_result(mode, "lane_change_unknown")
    if int(getattr(inputs, "lane_change_state", 0)) != 0:
      self.reset()
      return _preview_result(mode, "lane_change")
    if bool(getattr(inputs, "left_blinker", False)) or bool(getattr(inputs, "right_blinker", False)):
      self.reset()
      return _preview_result(mode, "blinker")
    if getattr(inputs, "steering_pressed", False) is not False:
      self.reset()
      return _preview_result(mode, "driver_override")

    # Transient blockers: decay via _soft_release instead of one-frame removal.
    model_age_s = _finite_float(getattr(inputs, "model_age_s", float("inf")))
    if model_age_s is None or model_age_s > LATERAL_PREVIEW_ASSIST_MAX_MODEL_AGE_S:
      self.reset()
      return _preview_result(mode, "model_stale")
    if bool(getattr(inputs, "steer_limited", False)):
      self._steer_limited_frames += 1
    else:
      self._steer_limited_frames = 0
    if self._steer_limited_frames >= self._steer_limited_block_frames:
      return self._soft_release(mode, "steer_limited", baseline_curvature_f, v_ego)
    if bool(getattr(inputs, "curvature_limited", False)):
      return self._soft_release(mode, "curvature_limited", baseline_curvature_f, v_ego)

    if model_path_result is None:
      self.reset()
      return _preview_result(mode, "invalid")
    # Model-path gating is a transient blocker, not driver intent: the quality gate flickers
    # in corners (routes 2cd/2ce: gated on 64.6% of corner jerk excursions vs 0.9% baseline,
    # mostly low_lane_confidence). Dropping here reset the nudge from its saturated cap to
    # zero in one frame 2427 times over two routes — |ay_delta| at the drop-out frame was
    # 0.05 m/s^2 at median, p99 and max, i.e. every drop-out was a full-cap step. Release it
    # like the other transient blockers instead; _soft_release only ever decays toward zero.
    if bool(getattr(model_path_result, "gated", False)):
      reason = str(getattr(model_path_result, "reason", "gated"))
      return self._soft_release(mode, reason, baseline_curvature_f, v_ego)
    if str(getattr(model_path_result, "reason", "invalid")) != "ok":
      reason = str(getattr(model_path_result, "reason", "invalid"))
      return self._soft_release(mode, reason, baseline_curvature_f, v_ego)
    if bool(getattr(model_path_result, "straight_path_stabilization_applied", False)):
      return self._soft_release(mode, "straight_path_stabilization", baseline_curvature_f, v_ego)

    path_quality = _finite_float(getattr(model_path_result, "quality", None))
    if path_quality is None:
      return self._soft_release(mode, "invalid", baseline_curvature_f, v_ego)
    confidence = float(np.clip(path_quality, 0.0, 1.0))
    if confidence < LATERAL_PREVIEW_ASSIST_MIN_PATH_QUALITY:
      return self._soft_release(mode, "low_quality", baseline_curvature_f, v_ego)

    lat_delay = _finite_float(getattr(inputs, "lat_delay", 0.0))
    if lat_delay is None or lat_delay < 0.0:
      return self._soft_release(mode, "invalid", baseline_curvature_f, v_ego)

    comfort_horizon = float(np.interp(v_ego, LATERAL_PREVIEW_ASSIST_COMFORT_HORIZON_BP, LATERAL_PREVIEW_ASSIST_COMFORT_HORIZON_S))
    t_preview = float(np.clip(lat_delay + comfort_horizon, LATERAL_PREVIEW_ASSIST_T_PREVIEW_MIN_S, LATERAL_PREVIEW_ASSIST_T_PREVIEW_MAX_S))

    preview_curvature = self._preview_curvature(inputs, t_preview, v_ego)
    if preview_curvature is None:
      return self._soft_release(mode, "invalid", baseline_curvature_f, v_ego)

    speed_sq = max(v_ego * v_ego, 1.0)
    ay_base = baseline_curvature_f * speed_sq
    ay_preview = preview_curvature * speed_sq
    if not math.isfinite(ay_base) or not math.isfinite(ay_preview):
      return self._soft_release(mode, "invalid", baseline_curvature_f, v_ego)

    if (abs(ay_base) >= LATERAL_PREVIEW_ASSIST_SIGN_CONFLICT_AY and abs(ay_preview) >= LATERAL_PREVIEW_ASSIST_SIGN_CONFLICT_AY
        and math.copysign(1.0, ay_base) != math.copysign(1.0, ay_preview)):
      # Direction reversals need fresh authority; never carry a stale opposite-sign nudge
      # through the turn transition.
      self.reset()
      return _preview_result(mode, "sign_conflict")

    ay_delta = float(np.clip(ay_preview - ay_base, -LATERAL_PREVIEW_ASSIST_MAX_DELTA_AY, LATERAL_PREVIEW_ASSIST_MAX_DELTA_AY))
    if not math.isfinite(ay_delta):
      return self._soft_release(mode, "invalid", baseline_curvature_f, v_ego)

    gain = float(np.interp(v_ego, LATERAL_PREVIEW_ASSIST_GAIN_BP, LATERAL_PREVIEW_ASSIST_GAIN))
    target_ay_delta = gain * confidence * ay_delta
    if not math.isfinite(target_ay_delta):
      return self._soft_release(mode, "invalid", baseline_curvature_f, v_ego)

    output_ay_delta, slew_limited = self._slew_output(target_ay_delta)
    curvature_nudge = output_ay_delta / speed_sq
    applied = mode == "apply" and abs(curvature_nudge) > 0.0
    return PreviewAssistResult(
      mode=mode,
      active=True,
      applied=applied,
      reason="ok",
      confidence=confidence,
      t_preview=t_preview,
      base_curvature=baseline_curvature_f,
      preview_curvature=preview_curvature,
      curvature_nudge=curvature_nudge,
      ay_base=ay_base,
      ay_preview=ay_preview,
      ay_delta=ay_delta,
      slew_limited=slew_limited,
    )
