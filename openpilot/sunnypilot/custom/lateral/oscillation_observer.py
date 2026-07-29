"""Shadow-only observer for lateral oscillation.

Classifies alternating torque/error sign reversals while the vehicle is near-straight,
engaged, and not limited/override. Output is logged to
``pid_log.adaptiveTorqueState.oscillationClassification`` but is never used for control.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

OSCILLATION_NONE = 0
OSCILLATION_MILD = 1
OSCILLATION_MODERATE = 2
OSCILLATION_SEVERE = 3

MIN_V_EGO = 10.0                    # m/s
MAX_DESIRED_LAT_ACCEL = 0.35        # m/s^2
MAX_STEERING_RATE = 100.0           # deg/s
MAX_TORQUE_FRAC = 0.95              # reject saturated output
WINDOW_TIME = 4.0                   # s, catches slower 0.5-1 Hz ping-pong seen in route logs
TORQUE_EPS = 0.05                   # to ignore noise in alternating sign
ERROR_EPS = 0.03                    # m/s^2, to ignore noise in alternating sign
MAX_CURRENT_ERROR_AMP = 0.45        # m/s^2, instant escape hatch
MAX_AVG_ERROR_AMP = 0.35            # m/s^2, small average error required
MIN_REVERSALS_MILD = 2              # sign reversals in window; one full alternating cycle
MIN_REVERSALS_MODERATE = 4
MIN_REVERSALS_SEVERE = 6


@dataclass
class OscillationDebug:
  classification: int = OSCILLATION_NONE
  reversals: int = 0
  avg_error_amp: float = 0.0
  gated: bool = False


def _sign(value: float, eps: float) -> int:
  if value > eps:
    return 1
  if value < -eps:
    return -1
  return 0


def _finite(*vals: float) -> bool:
  try:
    return all(math.isfinite(float(v)) for v in vals)
  except (TypeError, ValueError):
    return False


class OscillationObserver:
  """Stateful observer that counts sign reversals in a sliding window."""

  def __init__(self, dt: float):
    self.dt = max(float(dt), 1e-6)
    self.window_len = max(2, int(WINDOW_TIME / self.dt))
    self.torque_history: deque[int] = deque(maxlen=self.window_len)
    self.error_history: deque[int] = deque(maxlen=self.window_len)
    self.error_amp_history: deque[float] = deque(maxlen=self.window_len)
    self.last_debug = OscillationDebug()

  def reset(self) -> OscillationDebug:
    self.torque_history.clear()
    self.error_history.clear()
    self.error_amp_history.clear()
    self.last_debug = OscillationDebug()
    return self.last_debug

  def _count_reversals(self, history: deque[int]) -> int:
    if len(history) < 3:
      return 0
    vals = list(history)
    reversals = 0
    last_nz = vals[0]
    for s in vals[1:]:
      if s != 0:
        if last_nz != 0 and s != last_nz:
          reversals += 1
        last_nz = s
    return reversals

  def update(self, *, active: bool, v_ego: float, steering_pressed: bool,
             steer_limited_by_safety: bool, curvature_limited: bool,
             output_torque: float, steer_max: float,
             desired_lateral_accel: float, actual_lateral_accel: float,
             steering_rate_deg: float) -> OscillationDebug:
    if not _finite(v_ego, output_torque, steer_max, desired_lateral_accel,
                   actual_lateral_accel, steering_rate_deg):
      self.reset()
      self.last_debug = OscillationDebug(gated=True)
      return self.last_debug

    if (not active or steering_pressed or steer_limited_by_safety or
        curvature_limited or steer_max <= 0.0 or
        v_ego < MIN_V_EGO or
        abs(desired_lateral_accel) > MAX_DESIRED_LAT_ACCEL or
        abs(steering_rate_deg) > MAX_STEERING_RATE or
        abs(output_torque) >= MAX_TORQUE_FRAC * abs(steer_max)):
      self.torque_history.clear()
      self.error_history.clear()
      self.error_amp_history.clear()
      self.last_debug = OscillationDebug(gated=True)
      return self.last_debug

    torque_sign = _sign(output_torque, TORQUE_EPS)
    error = float(desired_lateral_accel - actual_lateral_accel)
    if abs(error) > MAX_CURRENT_ERROR_AMP:
      self.torque_history.clear()
      self.error_history.clear()
      self.error_amp_history.clear()
      self.last_debug = OscillationDebug(gated=True)
      return self.last_debug

    error_sign = _sign(error, ERROR_EPS)

    self.torque_history.append(torque_sign)
    self.error_history.append(error_sign)
    self.error_amp_history.append(abs(error))

    torque_reversals = self._count_reversals(self.torque_history)
    error_reversals = self._count_reversals(self.error_history)
    reversals = min(torque_reversals, error_reversals)

    avg_error_amp = (sum(self.error_amp_history) / len(self.error_amp_history)
                     if self.error_amp_history else 0.0)

    if avg_error_amp > MAX_AVG_ERROR_AMP:
      classification = OSCILLATION_NONE
    elif reversals >= MIN_REVERSALS_SEVERE:
      classification = OSCILLATION_SEVERE
    elif reversals >= MIN_REVERSALS_MODERATE:
      classification = OSCILLATION_MODERATE
    elif reversals >= MIN_REVERSALS_MILD:
      classification = OSCILLATION_MILD
    else:
      classification = OSCILLATION_NONE

    self.last_debug = OscillationDebug(
      classification=classification,
      reversals=reversals,
      avg_error_amp=avg_error_amp,
      gated=False,
    )
    return self.last_debug
