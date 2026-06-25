# distutils: language = c++
# cython: language_level = 3
import bisect
import math

from openpilot.sunnypilot.custom.lateral._output_governor_constants import (
  HIGH_RATE_START_DEG,
  ISO_ACCEL_MARGIN,
  ISO_LATERAL_ACCEL,
  NEAR_ISO_ACCEL_CAP,
  OVER_ISO_ACCEL_CAP,
  OVER_RESPONSE_FULL_EXCESS,
  OVER_RESPONSE_MARGIN,
  OVER_RESPONSE_MIN_SCALE,
  SIGN_CONFLICT_CAP,
  SIGN_THRESHOLD,
  STEERING_RATE_COMFORT_FULL_DEG,
  STEERING_RATE_COMFORT_MIN_CAP,
  STEERING_RATE_COMFORT_MIN_SLEW_SCALE,
  STEERING_RATE_COMFORT_START_DEG,
  TRACKING_CORRECTION_MARGIN,
  UNDER_RESPONSE_FADE_SPEED,
  UNDER_RESPONSE_FULL_SPEED,
  UNDER_RESPONSE_MARGIN,
  UNDER_RESPONSE_MAX_TORQUE_FRACTION,
)


def sign(value):
  if value > 0.0:
    return 1.0
  if value < 0.0:
    return -1.0
  return 0.0


def approach(value, target, step):
  if target > value:
    return min(target, value + step)
  return max(target, value - step)


def _finite(*values):
  for v in values:
    try:
      if not math.isfinite(float(v)):
        return False
    except (TypeError, ValueError):
      return False
  return True


def _interp(x, xp, fp):
  if x <= xp[0]:
    return float(fp[0])
  if x >= xp[-1]:
    return float(fp[-1])

  idx = bisect.bisect_right(xp, x) - 1
  if idx < 0:
    idx = 0

  x0 = float(xp[idx])
  x1 = float(xp[idx + 1])
  denom = x1 - x0
  if denom == 0.0:
    return float(fp[idx])

  y0 = float(fp[idx])
  y1 = float(fp[idx + 1])
  return y0 + (float(x) - x0) * (y1 - y0) / denom


def _clip(value, lower, upper):
  return max(lower, min(value, upper))


def _over_response_scale(inp):
  desired_sign = sign(inp.desired_lateral_accel)
  actual_sign = sign(inp.actual_lateral_accel)
  torque_sign = sign(inp.nominal_torque)
  if desired_sign == 0.0 or actual_sign != desired_sign or torque_sign != actual_sign:
    return 1.0
  over_response = desired_sign * (inp.actual_lateral_accel - inp.desired_lateral_accel)
  if over_response <= OVER_RESPONSE_MARGIN:
    return 1.0
  span = OVER_RESPONSE_FULL_EXCESS - OVER_RESPONSE_MARGIN
  ratio = _clip((over_response - OVER_RESPONSE_MARGIN) / max(span, 1e-3), 0.0, 1.0)
  return 1.0 + ratio * (OVER_RESPONSE_MIN_SCALE - 1.0)


def _sign_conflict(inp):
  desired_sign = sign(inp.desired_lateral_accel)
  actual_sign = sign(inp.actual_lateral_accel)
  return (desired_sign != 0.0 and actual_sign != 0.0 and desired_sign != actual_sign
          and abs(inp.actual_lateral_accel) > SIGN_THRESHOLD)


def _iso_cap(inp):
  actual_abs = abs(inp.actual_lateral_accel)
  output_reinforces_actual = (sign(inp.nominal_torque) != 0.0 and
                              sign(inp.nominal_torque) == sign(inp.actual_lateral_accel))
  if not output_reinforces_actual or actual_abs <= ISO_ACCEL_MARGIN:
    return 1.0
  return OVER_ISO_ACCEL_CAP if actual_abs > ISO_LATERAL_ACCEL else NEAR_ISO_ACCEL_CAP


def _under_response_floor(inp):
  if inp.v_ego >= UNDER_RESPONSE_FADE_SPEED:
    return 0.0
  desired_sign = sign(inp.desired_lateral_accel)
  actual_sign = 0.0 if abs(inp.actual_lateral_accel) <= SIGN_THRESHOLD else sign(inp.actual_lateral_accel)
  output_sign = sign(inp.nominal_torque)
  under_response = desired_sign * (inp.desired_lateral_accel - inp.actual_lateral_accel)
  same_sign_lag = desired_sign != 0.0 and output_sign == desired_sign and actual_sign in (0.0, desired_sign)
  corrective_reversal = (desired_sign != 0.0 and actual_sign != 0.0 and actual_sign != desired_sign
                         and output_sign == desired_sign)
  if under_response <= UNDER_RESPONSE_MARGIN or not (same_sign_lag or corrective_reversal):
    return 0.0
  if inp.v_ego <= UNDER_RESPONSE_FULL_SPEED:
    return 1.0
  span = UNDER_RESPONSE_FADE_SPEED - UNDER_RESPONSE_FULL_SPEED
  return _clip((UNDER_RESPONSE_FADE_SPEED - inp.v_ego) / max(span, 1e-3), 0.0, 1.0)


def _tracking_correction_needed(inp):
  output_sign = sign(inp.nominal_torque)
  lateral_accel_error = inp.desired_lateral_accel - inp.actual_lateral_accel
  if abs(lateral_accel_error) > TRACKING_CORRECTION_MARGIN and output_sign == sign(lateral_accel_error):
    return True

  desired_sign = sign(inp.desired_lateral_accel)
  actual_sign = 0.0 if abs(inp.actual_lateral_accel) <= SIGN_THRESHOLD else sign(inp.actual_lateral_accel)
  under_response = desired_sign * (inp.desired_lateral_accel - inp.actual_lateral_accel)
  same_sign_lag = desired_sign != 0.0 and output_sign == desired_sign and actual_sign in (0.0, desired_sign)
  corrective_reversal = (desired_sign != 0.0 and actual_sign != 0.0 and actual_sign != desired_sign
                         and output_sign == desired_sign)
  return under_response > UNDER_RESPONSE_MARGIN and (same_sign_lag or corrective_reversal)


def _steering_rate_comfort_blend(inp):
  torque_sign = sign(inp.nominal_torque)
  steering_rate_sign = sign(inp.steering_rate_deg)
  if torque_sign == 0.0 or torque_sign != steering_rate_sign:
    return 0.0
  if inp.release_active or _tracking_correction_needed(inp):
    return 0.0
  return _clip((abs(inp.steering_rate_deg) - STEERING_RATE_COMFORT_START_DEG) /
               max(STEERING_RATE_COMFORT_FULL_DEG - STEERING_RATE_COMFORT_START_DEG, 1e-3), 0.0, 1.0)
