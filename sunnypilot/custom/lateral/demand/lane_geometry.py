"""Lane-line geometry helper for inner-lane advisory centering.

Phase 1: uses only modelV2.laneLines[1] and [2] (inner lane lines). Road edges are
explicitly out of scope. All outputs are fail-closed: malformed or borderline input
returns a geometry object with valid=False and a reason string.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


# Sample points for inner-lane geometry (meters ahead).
LANE_GEOMETRY_SAMPLE_XS = (10.0, 20.0, 30.0)

# Quality thresholds. Intentionally conservative for phase 1 advisory use.
LANE_GEOMETRY_MIN_INNER_PROB = 0.85
LANE_GEOMETRY_MAX_INNER_STD = 0.25
LANE_GEOMETRY_MIN_LANE_WIDTH = 2.0
LANE_GEOMETRY_MAX_LANE_WIDTH = 6.0
LANE_GEOMETRY_WIDTH_MAX_SPREAD_RATIO = 0.25  # max |w_near - w_preview| / mean(widths)
LANE_GEOMETRY_SIGN_AGREEMENT_THRESHOLD = 0.05  # m; offsets must agree in sign outside this
LANE_GEOMETRY_MIN_STRADDLE_MARGIN = 0.10  # m; both inner lines must bracket ego with margin

LANE_GEOMETRY_REASON_OK = "ok"
LANE_GEOMETRY_REASON_MISSING = "missing_lanes"
LANE_GEOMETRY_REASON_LOW_PROB = "low_prob"
LANE_GEOMETRY_REASON_HIGH_STD = "high_std"
LANE_GEOMETRY_REASON_BAD_WIDTH = "bad_width"
LANE_GEOMETRY_REASON_UNSTABLE_WIDTH = "unstable_width"
LANE_GEOMETRY_REASON_SIGN_MISMATCH = "sign_mismatch"
LANE_GEOMETRY_REASON_BAD_STRADDLE = "bad_straddle"
LANE_GEOMETRY_REASON_BAD_PATH = "bad_path"
LANE_GEOMETRY_REASON_BAD_GEOMETRY = "bad_geometry"

# Single-confident-line geometry (route 0000025a drift diagnosis): both-line geometry is
# unavailable in exactly the urban regime where slow off-center drift happens (one inner
# line confident, the other dead). One confident line + a recently-learned lane width
# still defines lane center. Fail-closed like the both-line path.
LANE_GEOMETRY_ONE_LINE_MIN_ABS_Y = 0.5   # m; line must be plausibly beside ego, not under it
LANE_GEOMETRY_ONE_LINE_MAX_ABS_Y = 4.0   # m; beyond this it is an adjacent-lane line
LANE_GEOMETRY_ONE_LINE_CONFIDENCE_DISCOUNT = 0.7  # width is remembered, not observed
LANE_GEOMETRY_REASON_NO_LEARNED_WIDTH = "no_learned_width"
LANE_GEOMETRY_REASON_AMBIGUOUS_LINES = "ambiguous_lines"
LANE_GEOMETRY_REASON_BAD_LINE_SIDE = "bad_line_side"


@dataclass(frozen=True)
class LaneGeometryResult:
  valid: bool
  reason: str
  confidence: float
  source: str
  lateral_error: float  # lane_center_y - model_y at near sample
  predicted_lateral_error: float  # lane_center_y - model_y at preview sample
  heading_error: float  # model heading at near sample (kept from path, not geometry)
  width_near: float
  width_preview: float
  offset_near: float
  offset_preview: float
  lane_center_y_near: float
  lane_center_y_preview: float
  model_y_near: float
  model_y_preview: float
  prob_left: float
  prob_right: float
  std_left: float
  std_right: float


def _finite(value: Any) -> float | None:
  try:
    f = float(value)
  except (TypeError, ValueError):
    return None
  return f if math.isfinite(f) else None


def _lane_line_y_at(lane_line: Any, x: float) -> float | None:
  raw_xs = getattr(lane_line, "x", None) or ()
  raw_ys = getattr(lane_line, "y", None) or ()
  xs = _finite_array(raw_xs)
  ys = _finite_array(raw_ys)
  if len(xs) < 2 or len(xs) != len(ys):
    return None
  x0, x1 = xs[0], xs[-1]
  # Fail closed if the requested sample is outside the lane-line's own horizon.
  if x < x0 or x > x1:
    return None
  return float(np.interp(x, xs, ys))


def _path_y_at(position_x: Sequence[float], position_y: Sequence[float], x: float) -> float | None:
  xs = _finite_array(position_x)
  ys = _finite_array(position_y)
  if len(xs) < 2 or len(xs) != len(ys) or xs[-1] <= xs[0]:
    return None
  if x < xs[0] or x > xs[-1]:
    return None
  return float(np.interp(x, xs, ys))


def _finite_array(values: Sequence[float]) -> list[float]:
  result: list[float] = []
  for value in values:
    f = _finite(value)
    if f is None:
      return []
    result.append(f)
  return result


def evaluate_lane_geometry(
  *,
  lane_lines: Sequence[Any],
  lane_line_probs: Sequence[float],
  lane_line_stds: Sequence[float],
  position_x: Sequence[float],
  position_y: Sequence[float],
  near_x: float,
  preview_x: float,
  model_y_near: float | None = None,
  model_y_preview: float | None = None,
) -> LaneGeometryResult:
  """Evaluate inner-lane geometry for advisory lane centering.

  Returns a LaneGeometryResult. ``valid`` is True only when both inner lane lines
  are present, high-confidence, low-std, produce a plausible/stable lane width, and
  the near/preview offsets agree in sign.

  Coordinate convention matches the modelV2 path: positive y is to the right of the
  vehicle. Therefore ``lateral_error = lane_center_y - model_y`` is positive when
  the path is left of the lane center and negative when the path is right of center.
  """
  source = "lane_lines"
  def failure(r: str) -> LaneGeometryResult:
    return LaneGeometryResult(
      valid=False, reason=r, confidence=0.0, source=source,
      lateral_error=0.0, predicted_lateral_error=0.0, heading_error=0.0,
      width_near=0.0, width_preview=0.0,
      offset_near=0.0, offset_preview=0.0,
      lane_center_y_near=0.0, lane_center_y_preview=0.0,
      model_y_near=0.0, model_y_preview=0.0,
      prob_left=0.0, prob_right=0.0, std_left=0.0, std_right=0.0,
    )

  if (lane_lines is None or len(lane_lines) < 4 or
      lane_line_probs is None or len(lane_line_probs) < 4 or
      lane_line_stds is None or len(lane_line_stds) < 4):
    return failure(LANE_GEOMETRY_REASON_MISSING)

  left_lane = lane_lines[1]
  right_lane = lane_lines[2]
  if left_lane is None or right_lane is None:
    return failure(LANE_GEOMETRY_REASON_MISSING)

  prob_l = _finite(lane_line_probs[1])
  prob_r = _finite(lane_line_probs[2])
  std_l = _finite(lane_line_stds[1])
  std_r = _finite(lane_line_stds[2])
  if prob_l is None or prob_r is None:
    return failure(LANE_GEOMETRY_REASON_LOW_PROB)
  if std_l is None or std_r is None:
    return failure(LANE_GEOMETRY_REASON_HIGH_STD)
  if prob_l < LANE_GEOMETRY_MIN_INNER_PROB or prob_r < LANE_GEOMETRY_MIN_INNER_PROB:
    return failure(LANE_GEOMETRY_REASON_LOW_PROB)
  if std_l > LANE_GEOMETRY_MAX_INNER_STD or std_r > LANE_GEOMETRY_MAX_INNER_STD:
    return failure(LANE_GEOMETRY_REASON_HIGH_STD)

  left_y_near = _lane_line_y_at(left_lane, near_x)
  right_y_near = _lane_line_y_at(right_lane, near_x)
  left_y_preview = _lane_line_y_at(left_lane, preview_x)
  right_y_preview = _lane_line_y_at(right_lane, preview_x)
  if any(v is None for v in (left_y_near, right_y_near, left_y_preview, right_y_preview)):
    return failure(LANE_GEOMETRY_REASON_MISSING)
  # Narrow types after the None check above.
  assert left_y_near is not None and right_y_near is not None
  assert left_y_preview is not None and right_y_preview is not None

  # Inner lane lines must bracket the ego/path frame. Same-side inner lines can be
  # adjacent-lane or off-lane geometry; never use them for active centering.
  if not (_straddles_ego(left_y_near, right_y_near) and _straddles_ego(left_y_preview, right_y_preview)):
    return failure(LANE_GEOMETRY_REASON_BAD_STRADDLE)

  width_near = abs(left_y_near - right_y_near)
  width_preview = abs(left_y_preview - right_y_preview)
  if not (LANE_GEOMETRY_MIN_LANE_WIDTH <= width_near <= LANE_GEOMETRY_MAX_LANE_WIDTH and
          LANE_GEOMETRY_MIN_LANE_WIDTH <= width_preview <= LANE_GEOMETRY_MAX_LANE_WIDTH):
    return failure(LANE_GEOMETRY_REASON_BAD_WIDTH)

  mean_width = (width_near + width_preview) / 2.0
  if mean_width > 0.0:
    width_spread = abs(width_near - width_preview) / mean_width
  else:
    width_spread = float("inf")
  if width_spread > LANE_GEOMETRY_WIDTH_MAX_SPREAD_RATIO:
    return failure(LANE_GEOMETRY_REASON_UNSTABLE_WIDTH)

  center_y_near = (float(left_y_near) + float(right_y_near)) * 0.5
  center_y_preview = (float(left_y_preview) + float(right_y_preview)) * 0.5

  if model_y_near is None or model_y_preview is None:
    model_y_near = _path_y_at(position_x, position_y, near_x)
    model_y_preview = _path_y_at(position_x, position_y, preview_x)
  else:
    model_y_near = _finite(model_y_near)
    model_y_preview = _finite(model_y_preview)
  if model_y_near is None or model_y_preview is None:
    return failure(LANE_GEOMETRY_REASON_BAD_PATH)

  offset_near = center_y_near - model_y_near
  offset_preview = center_y_preview - model_y_preview

  # Near/preview sign agreement: both pointing the same direction relative to center.
  if abs(offset_near) > LANE_GEOMETRY_SIGN_AGREEMENT_THRESHOLD and abs(offset_preview) > LANE_GEOMETRY_SIGN_AGREEMENT_THRESHOLD:
    if math.copysign(1.0, offset_near) != math.copysign(1.0, offset_preview):
      return failure(LANE_GEOMETRY_REASON_SIGN_MISMATCH)

  if not all(math.isfinite(v) for v in (offset_near, offset_preview, center_y_near, center_y_preview)):
    return failure(LANE_GEOMETRY_REASON_BAD_GEOMETRY)

  # Confidence blends lane-line probability and inverse std.
  prob_confidence = min(prob_l, prob_r)
  std_confidence = max(0.0, 1.0 - (max(std_l, std_r) / LANE_GEOMETRY_MAX_INNER_STD))
  width_confidence = max(0.0, 1.0 - width_spread / LANE_GEOMETRY_WIDTH_MAX_SPREAD_RATIO)
  confidence = min(prob_confidence, std_confidence, width_confidence)

  return LaneGeometryResult(
    valid=True,
    reason=LANE_GEOMETRY_REASON_OK,
    confidence=float(np.clip(confidence, 0.0, 1.0)),
    source=source,
    lateral_error=offset_near,
    predicted_lateral_error=offset_preview,
    heading_error=0.0,
    width_near=width_near,
    width_preview=width_preview,
    offset_near=offset_near,
    offset_preview=offset_preview,
    lane_center_y_near=center_y_near,
    lane_center_y_preview=center_y_preview,
    model_y_near=model_y_near,
    model_y_preview=model_y_preview,
    prob_left=prob_l,
    prob_right=prob_r,
    std_left=std_l,
    std_right=std_r,
  )


def _straddles_ego(left_y: float, right_y: float) -> bool:
  return left_y <= -LANE_GEOMETRY_MIN_STRADDLE_MARGIN and right_y >= LANE_GEOMETRY_MIN_STRADDLE_MARGIN


def evaluate_single_line_geometry(
  *,
  lane_lines: Sequence[Any],
  lane_line_probs: Sequence[float],
  lane_line_stds: Sequence[float],
  position_x: Sequence[float],
  position_y: Sequence[float],
  near_x: float,
  preview_x: float,
  learned_width: float,
  model_y_near: float | None = None,
  model_y_preview: float | None = None,
) -> LaneGeometryResult:
  """Lane-center geometry from exactly ONE confident inner line + a learned lane width.

  Intended for the regime where both-line geometry is unavailable because one inner
  line is dead. Fail-closed: requires an unambiguous single confident line (the other
  inner line must be below the prob threshold), the line on the side its index claims,
  and a plausible recently-learned width supplied by the caller. Same coordinate
  convention as evaluate_lane_geometry (positive y right of vehicle; left inner line
  has negative y). Confidence is discounted because width is remembered, not observed.
  """
  source = "single_line"
  def failure(r: str) -> LaneGeometryResult:
    return LaneGeometryResult(
      valid=False, reason=r, confidence=0.0, source=source,
      lateral_error=0.0, predicted_lateral_error=0.0, heading_error=0.0,
      width_near=0.0, width_preview=0.0,
      offset_near=0.0, offset_preview=0.0,
      lane_center_y_near=0.0, lane_center_y_preview=0.0,
      model_y_near=0.0, model_y_preview=0.0,
      prob_left=0.0, prob_right=0.0, std_left=0.0, std_right=0.0,
    )

  width = _finite(learned_width)
  if width is None or not (LANE_GEOMETRY_MIN_LANE_WIDTH <= width <= LANE_GEOMETRY_MAX_LANE_WIDTH):
    return failure(LANE_GEOMETRY_REASON_NO_LEARNED_WIDTH)

  if (lane_lines is None or len(lane_lines) < 4 or
      lane_line_probs is None or len(lane_line_probs) < 4 or
      lane_line_stds is None or len(lane_line_stds) < 4):
    return failure(LANE_GEOMETRY_REASON_MISSING)

  candidates = []
  for idx in (1, 2):
    prob = _finite(lane_line_probs[idx])
    std = _finite(lane_line_stds[idx])
    if (prob is not None and std is not None and
        prob >= LANE_GEOMETRY_MIN_INNER_PROB and std <= LANE_GEOMETRY_MAX_INNER_STD and
        lane_lines[idx] is not None):
      candidates.append((idx, prob, std))
  if len(candidates) != 1:
    # Zero confident lines: nothing to anchor to. Two confident lines: the both-line
    # path should own it; if it declined (bad width/straddle) the lines are suspect.
    return failure(LANE_GEOMETRY_REASON_LOW_PROB if not candidates else LANE_GEOMETRY_REASON_AMBIGUOUS_LINES)

  idx, prob, std = candidates[0]
  line_y_near = _lane_line_y_at(lane_lines[idx], near_x)
  line_y_preview = _lane_line_y_at(lane_lines[idx], preview_x)
  if line_y_near is None or line_y_preview is None:
    return failure(LANE_GEOMETRY_REASON_MISSING)

  # The line must sit on the side its index claims (left inner y<0, right inner y>0)
  # and at a plausible in-lane distance; otherwise it is an adjacent-lane or off-lane line.
  side = -1.0 if idx == 1 else 1.0
  for y in (line_y_near, line_y_preview):
    if not (LANE_GEOMETRY_ONE_LINE_MIN_ABS_Y <= side * y <= LANE_GEOMETRY_ONE_LINE_MAX_ABS_Y):
      return failure(LANE_GEOMETRY_REASON_BAD_LINE_SIDE)

  half_width = width * 0.5
  # Right inner line = left inner line + width in y, so center is line -/+ half width.
  center_y_near = line_y_near - side * half_width
  center_y_preview = line_y_preview - side * half_width

  if model_y_near is None or model_y_preview is None:
    model_y_near = _path_y_at(position_x, position_y, near_x)
    model_y_preview = _path_y_at(position_x, position_y, preview_x)
  else:
    model_y_near = _finite(model_y_near)
    model_y_preview = _finite(model_y_preview)
  if model_y_near is None or model_y_preview is None:
    return failure(LANE_GEOMETRY_REASON_BAD_PATH)

  offset_near = center_y_near - model_y_near
  offset_preview = center_y_preview - model_y_preview
  if abs(offset_near) > LANE_GEOMETRY_SIGN_AGREEMENT_THRESHOLD and abs(offset_preview) > LANE_GEOMETRY_SIGN_AGREEMENT_THRESHOLD:
    if math.copysign(1.0, offset_near) != math.copysign(1.0, offset_preview):
      return failure(LANE_GEOMETRY_REASON_SIGN_MISMATCH)
  if not all(math.isfinite(v) for v in (offset_near, offset_preview, center_y_near, center_y_preview)):
    return failure(LANE_GEOMETRY_REASON_BAD_GEOMETRY)

  std_confidence = max(0.0, 1.0 - (std / LANE_GEOMETRY_MAX_INNER_STD))
  confidence = min(prob, std_confidence) * LANE_GEOMETRY_ONE_LINE_CONFIDENCE_DISCOUNT

  return LaneGeometryResult(
    valid=True,
    reason=LANE_GEOMETRY_REASON_OK,
    confidence=float(np.clip(confidence, 0.0, 1.0)),
    source=source,
    lateral_error=offset_near,
    predicted_lateral_error=offset_preview,
    heading_error=0.0,
    width_near=width,
    width_preview=width,
    offset_near=offset_near,
    offset_preview=offset_preview,
    lane_center_y_near=center_y_near,
    lane_center_y_preview=center_y_preview,
    model_y_near=model_y_near,
    model_y_preview=model_y_preview,
    prob_left=prob if idx == 1 else 0.0,
    prob_right=prob if idx == 2 else 0.0,
    std_left=std if idx == 1 else 0.0,
    std_right=std if idx == 2 else 0.0,
  )
