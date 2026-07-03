# Rate-limit straight-path stabilization releases and detect entry trends by EMA

Status: accepted
Date: 2026-07-03
Relates to: `sunnypilot/custom/lateral/demand/model_path_processor.py` (straight-path stabilization).

## Context

Straight-path stabilization (SPS, `apply` mode since 2026-07-01) anchors near-straight demand to a
rolling median and clips deviation to ±0.02 m/s². The anchor buffer deliberately appends the
clipped candidate (not raw) so it does not chase the wobble it suppresses — but that also means it
cannot follow a slow curve-entry ramp. The rising-frames release (3 consecutive same-sign frames
summing > 0.15 m/s²) was defeated by frame-to-frame model noise, so on gentle curve entries
suppression grew until `|a_raw − anchor| > 0.35 m/s²` and `release_suppression` returned the raw
target in a single frame.

Route `0000024b--796aeaea89` (2026-07-03): those one-frame releases stepped demand by up to
~0.45 m/s² (≈45 m/s³), amplified by the response core's jerk-anticipation setpoint term, and were
the dominant felt "sudden lateral spike jerk" (4 dissected events; 80 post-applied step episodes
in a full-route replay).

## Decision

1. **Rate-limit all SPS output transitions** (`_slew_sps_transitions`): after SPS stops applying —
   release, pause, or re-apply — the output blends toward the passthrough target at
   `SPS_RELEASE_RAMP_LAT_JERK = 1.2 m/s³` in lat-accel space instead of stepping. The blend state
   survives the release's anchor reset but is cleared by the pipeline's full SPS resets
   (terminal fallbacks), so a stale blend can never yank demand after a fallback window.
2. **Replace the rising-frames detector with a winsorized-EMA trend release**: per-frame raw
   lat-accel slope clipped to ±1.5 m/s³ feeds an EMA (τ = 0.3 s); `release_trend` fires above
   0.25 m/s³. A single-frame model jump contributes ≤ ~0.05 (stays suppressed — preserves the
   post-dropout redetect behavior); a sustained ~0.3 m/s³ curve entry releases in ~0.5 s, well
   before the 0.35 suppression cap. Constants calibrated on route `0000024b` rlogs (straight-cruise
   |EMA| p99 ≈ 0.23; entry peaks 0.18–0.33).

`SPS_MAX_SUPPRESSION_LAT_ACCEL` stays 0.35: with releases rate-limited it is no longer a jerk
source, and lowering it would weaken suppression of post-dropout model blips.

## Consequences

- Route-replay A/B (22.1 min active): post-applied one-frame demand steps > 0.15 m/s² drop from
  80 episodes (max 0.42) to 1 (max 0.22, a raw pass-through jump); applied duty unchanged
  (23.1% → 23.2%); `release_trend` fires ~0.7/min, each now blended.
- The no-sign-flip-vs-raw invariant is transiently relaxed during a release blend: an opposite-sign
  tail decays at the ramp rate (< 0.1 s for typical magnitudes). Pinned in
  `test_sps_no_sign_flip_except_tiny_raw`.
- Demand jerk smoothing was evaluated for the residual raw model-path jump family and left
  unchanged: it is already force-enabled in wiring and effective where eligible (route 24b: the two
  eligible jumps smoothed 0.16/0.19 → 0.01 m/s²); the excluded jumps are lane-change/blinker,
  mid-curve, or low-quality contexts where smoothing distrusted demand is deliberately avoided.
