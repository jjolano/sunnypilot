# SPS transition chatter and LCA availability fixes

Date: 2026-07-06
Status: accepted
Evidence: route `0000025e--8444db48bd` (2026-07-05, build `1869b2e1a8`), drive_lab
lateral gate verdict `torque_event_dominant` (torque 152.6 vs wander 23.6).

## Context

Felt wheel jerks and near-line drift on route 25e traced to four defects:

1. SPS paused 1,569 of 1,583 off-edges on `steer_limited` (controlsd's
   `steer_limited_by_safety`, |cmd − applied| > 0.01), which chatters at ~8 Hz on HKG
   torque ramps. SPS sawtoothed anchor↔raw at up to 7.6 toggles/s.
2. `_slew_sps_transitions` cleared its blend state once a release converged, so the
   next apply snapped demand to the anchor unramped: 42 one-frame steps > 0.1 m/s²,
   max 0.27 m/s² — 23 of the route's top 25 pipeline-injected demand steps.
3. LCA was hard-blocked by path reason `low_lane_confidence` 81.6% of active time
   (one weak lane line on city roads — exactly the regime one-line centering exists
   for), active during only 46.5% of time spent within 1.1 m of a line.
4. LCA's flat 0.08 m/s² authority cap saturated during the one drift recovery it
   attempted (t≈364–376 s) while error grew 0 → 0.7 m until driver override.

## Decisions

1. **Delete the `gate_steer_limited` SPS pause.** It shipped with the original SPS
   commit undocumented; pausing suppression on a torque-transport symptom reintroduces
   raw wobble exactly when the actuator is fighting, and the divergence/trend/large-accel
   releases already cover real runaway.
2. **Track demand while SPS is inactive** in `_slew_sps_transitions` so every
   transition, both directions, is bounded by `SPS_RELEASE_RAMP_LAT_JERK`. Converged
   passthrough stays transparent (no global jerk limit on raw demand).
3. **Allow path reason `low_lane_confidence` through LCA's gates** (hard gate, cooldown
   triggers, relax quality checks) via `LANE_CENTERING_ASSIST_ALLOWED_PATH_REASONS`,
   matching SPS. Per-line probability gating already lives in `_confidence()` (needs
   min(left,right) prob ≥ 0.5 without geometry) and one-line/geometry validity.
4. **Escalate the LCA nudge cap with predicted drift**: interp from the unchanged base
   cap at ≤ 0.30 m predicted error to 0.18 m/s² at ≥ 0.60 m. Small-error behavior,
   including straight-cruise damping, is unchanged.

## Validation

Stage-level A/B replay of the SPS stage over route 25e's recorded raw curvature,
v_ego, quality/reason, and reconstructed `steer_limited` sequence
(old HEAD vs fixed): apply-edges 1,429 → 24; injected one-frame steps > 0.1 m/s²
58 → 3 (remaining are raw model jumps, not SPS transitions); > 0.2 m/s² 8 → 0;
max 0.279 → 0.133; applied duty 29% → 51%. Unit coverage:
`test_sps_ignores_steer_limited`,
`test_sps_transition_slew_bounds_reapply_and_keeps_passthrough_transparent`,
`test_lane_centering_assist_runs_under_low_lane_confidence`,
`test_lane_centering_nudge_cap_escalates_with_predicted_drift`.

## Risks / follow-up

- LCA now runs on low-lane-confidence frames with path quality ≥ 0.85; watch the next
  routes for center-chase wander regression (relax machinery and confidence scaling are
  the guards).
- Higher SPS applied duty (~51%) means more anchoring; trend/sign-flip releases still
  fire on curve entries — verify on a curvy route.
- `ModelPathProcessorInputs.steer_limited` is now unused by SPS but still plumbed
  through wiring; remove the dead plumbing once concurrent wiring work lands.
