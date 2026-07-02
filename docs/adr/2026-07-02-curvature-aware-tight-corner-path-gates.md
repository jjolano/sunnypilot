# Curvature-aware path gates for tight low-speed corners

Status: accepted
Date: 2026-07-02
Relates to: `sunnypilot/custom/lateral/demand/model_path_processor.py`.

## Context

City tight-corner drives (routes `00000240`/`00000242`/`00000241`, 2026-07) show 7–26 steering
overrides/min at 3–10 m/s with the driver consistently adding torque *into* the turn. The torque
controller is not the cause: output is never saturated and actual curvature tracks final desired
~1:1. The demand layer is: during tight-corner frames the path evidence quality drops to a median
of 0.45 and `_handle_low_quality` blends desired curvature toward the measured-curvature fallback
(plus the low-speed untrusted step limit of 0.0025/frame), which under-turns exactly at corner
entry — the driver fills the gap.

Measured against the gate thresholds on those routes (first 17 path points, |curv| > 0.02, v < 12):

- max path `yStd` median 1.5, p90 2.1 — exceeds `MAX_PATH_Y_STD = 0.8` in **91–93%** of frames.
  Path y-uncertainty scales with path bend; the threshold assumes near-straight paths.
- max |dy/dx| p90 1.2–1.5 — exceeds `MAX_CORE_PATH_LATERAL_SLOPE = 1.0` (45°) in **28–29%** of
  frames, tripping the *hard* `invalid_path` fallback. A planned 90° intersection turn
  legitimately exceeds 45° of lateral slope.

The gates protect against genuine model garbage on straights, so blanket threshold raises are out.

## Decision

Make both gates forgive legitimate tight turns only when the car is *measurably* turning the same
way at low speed:

1. Extend the existing `_low_speed_measured_turn_confirms_curvature` quality bump (previously only
   for `low_lane_confidence`) to `high_path_std`. The confirmation requires v < 12 m/s, matching
   signs, path disagreement ≤ 0.75 m/s², and desired-vs-measured lateral accel within 0.75 m/s².
2. Widen the core-path slope limit from 1.0 to up to 2.0, interpolated on |measured curvature|
   over [0.02, 0.06] 1/m, and only below 12 m/s (`_core_path_slope_limit`). With measured
   curvature near zero — i.e. on straights, where a steep path is garbage — the limit stays 1.0.

## Consequences

- Demand follows the model path through city corners once the turn is underway instead of
  clamping to lagging measured curvature; expected to cut the into-the-turn override pattern.
- Turn *entry* (before measured curvature builds) still passes through one or two gated frames;
  acceptable, and fail-closed by design.
- Straight-road protection is unchanged: both escapes require measured confirmation, so a model
  path spike on a straight still gates/invalidates exactly as before.
- `model_path_processor_v1` (frozen characterization twin) deliberately keeps the old behavior;
  parity sequences avoid confirmed-tight-turn inputs.
- Validation: re-run the override-rate/direction extraction on the next city route with
  intersection turns (see memory note `city-corner-demand-gating`); watch for any new
  wide-line/steep-path acceptance on straights in `lateral_event_report`.
