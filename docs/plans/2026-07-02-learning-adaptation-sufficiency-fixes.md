# Learning/adaptation sufficiency fixes

Follow-up to the 2026-07-02 assessment of whether the lateral learning strategy matches
the data we can actually collect. Verdict was: the offline rlog-replay evidence path is
sound; on-device convergence and two verdict gates are not. This plan fixes the
mismatches, smallest-diff-first. No new Params keys, no new telemetry, no apply-path
behavior changes — everything here is write-side learning logic, evidence collection
bounds, docs, and one small tool extension.

## Phase 1 — Cross-drive profile blending (highest value)

**Problem.** `TorqueEstimatorExt` buckets (`speed_learning_buckets`, `low_speed_buckets`,
`roll_comp_buckets`) start empty every ignition, and `maybe_persist_speed_profile`
overwrites the stored profile with whatever the current drive fit. Roll comp needs 2000
qualifying points in one ignition cycle; a marginal drive that barely crosses
`MIN_POINTS` clobbers a better fit from a long drive. The learned gain is effectively
"gain from the last long drive".

**Fix.** Blend at write time instead of persisting raw points (raw bucket persistence
would be ~180k floats of Params churn; blending is a ~30-line diff and the parsed old
profile is already in hand as `roll_comp_profile_cache` / `speed_profile_cache`, with
restore-key and version guards already enforced by the parsers).

In `sunnypilot/selfdrive/locationd/torqued_ext.py` `maybe_persist_speed_profile`:

- Roll comp: if `roll_comp_profile_cache` parsed valid, write
  `gain = (w_old * old_gain + w_new * new_gain) / (w_old + w_new)` with
  `w = min(points, 2 * MIN_POINTS)`; stored `points = min(old + new, 4 * MIN_POINTS)`
  (cap keeps old evidence decaying so adaptation never freezes — a mature profile still
  moves ≥ ~1/3 of the way toward a full new fit); `span = max(old, new)`;
  `confidence` recomputed from capped points. If the old profile failed parse
  (restore-key mismatch, version bump, corrupt), keep today's overwrite — that is the
  correct behavior there.
- Speed-aware: same rule per anchor, blending `ratios` only at anchors where **both**
  old and new confidence > 0; where only one side has confidence, take that side.
  Anchors are fixed `SPEED_BUCKET_BP`, so alignment is positional. Per-anchor `points`
  sum with the same style of cap; `confidence` recomputed. `lowSpeed` section stays
  last-write (evidence-only, runtime ignores it by construction).
- Blend helpers live next to the fitters in `roll_comp_learning.py` /
  `speed_aware_torque.py` so they're unit-testable without a `TorqueEstimatorExt`.
- Format version stays 1 — the on-disk schema is unchanged; blending is write-side only.
  Blended output must still satisfy its own parser (gain within
  `[ROLL_GAIN_MIN, ROLL_GAIN_MAX]`, points ≥ `MIN_POINTS`, span ≥ `MIN_X_SPAN`,
  confidence in `[MIN_CONFIDENCE, 1]`) — add an assert-style test for this round trip.

**Tests.** Extend `sunnypilot/custom/lateral/tests/test_roll_comp_learning.py` and
`test_speed_aware_torque.py` (blend math, cap behavior, single-sided confidence,
parse round-trip of blended output). Extend
`selfdrive/locationd/test/test_torqued_roll.py`: persist path blends with an existing
cache rather than overwriting; overwrites when cache restore-key mismatches.

## Phase 2 — Rewrite the roll-comp promote gate for a one-car fleet (doc-only)

**Problem.** `docs/plans/2026-07-02-shadow-feature-verdicts.md` gates roll-comp
promotion on "≥3 cars with meaningful crown". The fleet is one device; as written the
2026-08-15 deadline auto-verdicts to delete regardless of data quality.

**Fix.** Restate the gate in terms one car produces:

- **Promote** if ≥3 routes each have meaningful crown (per-route
  `roll_comp_profile` `roll_span` above a threshold, propose ≥ 0.3 m/s²), the learned
  gain spread across those routes is < 0.05, and replay shows straight-cruise tracking
  error no worse than fixed 0.55.
- **Park/Delete** clauses unchanged in spirit; per-platform generalization becomes a
  later concern if the fleet ever grows.

## Phase 3 — State the extrapolation assumption in the ADR (doc-only)

**Problem.** The gain is learned only in a narrow regime (straight road, steady state,
v > 15 m/s, |roll| ≤ 0.1 rad) but applied at all speeds and curvatures, including
banked curves the collection gates exclude by construction. The promote criterion
"no degradation on banked curves" can only ever be checked indirectly (replay tracking
error), never from learning evidence.

**Fix.** Add a "Limits" section to
`docs/adr/2026-07-02-learned-roll-compensation-gain.md`: linearity of the roll response
beyond ±0.1 rad is an assumption; banked-curve validation is replay-side only; the
learned gain must stay inside `[ROLL_GAIN_MIN, ROLL_GAIN_MAX]` precisely because the
learner cannot see the regimes where a bad extrapolation would hurt most.

## Phase 4 — Low-speed shadow bucket coverage

**Problem.** `low_speed_buckets` reuses the highway x-bounds (|steer| < 0.5) and the
collection gate caps |lateral_acc| at 2.5 m/s², so tight city corners demanding
near-full torque — the regime that motivated the buckets (city corner demand gating) —
are partially dropped at the door.

**Fix.** In `torqued_ext.py`: give `low_speed_buckets` its own x-bounds extending to
±1.0 (torque command saturates at 1.0; add outer buckets rather than stretching the
inner ones, e.g. append `(0.5, 1.0)` / `(-1.0, -0.5)`), and raise the low-speed
collection cap from 2.5 to 3.0 m/s². Safe because the low-speed section is
evidence-only: `fit_low_speed_section` is unclamped and reported, and runtime ratio
lookup ignores it by construction.

**Tests.** Extend the bucket-routing cases in
`selfdrive/locationd/test/test_torqued_shadow.py` (or the speed-aware unit tests) for
the new outer buckets and cap.

## Phase 5 — Cross-route roll-comp verdict tool

**Problem.** The runbook's roll-comp evidence source is "engaged-route replay" across
routes, but `tools/drive_lab/roll_comp_profile.py` only reports one route;
`speed_adaptive_verdict.py` has the multi-route + verdict pattern the gate needs.

**Fix.** Extend `roll_comp_profile.py` `main` to accept `ROUTE [ROUTE ...]` and, for
multiple routes, print per-route slope/span/points plus the Phase-2 gate verdict
(routes with span ≥ threshold, slope spread, promote/park/insufficient). Reuse
`build_roll_comp_profile` as-is; no new module.

## Explicitly not doing

- Raw bucket-point persistence across drives (Params churn; blending achieves the goal).
- New telemetry for the disturbance classifier (aggregate counters suffice — rlog replay
  re-derives per-event decisions).
- Any change to apply paths, safety clamps, or the parsers' fail-closed behavior.
- New Params keys or Settings UI work (no user-facing surface changes).

## Order and validation

1 → 4 are one commit each on `master` (Phase 2+3 can share a docs commit). Phase 5 last.
After Phase 1 lands, run:

```bash
uv run --extra testing --extra tools python -m pytest \
  sunnypilot/custom/lateral/tests/test_roll_comp_learning.py \
  sunnypilot/custom/lateral/tests/test_speed_aware_torque.py \
  selfdrive/locationd/test/test_torqued_roll.py \
  selfdrive/locationd/test/test_torqued_shadow.py
```

Device validation gates from the lateral learning program remain open and unchanged:
next shadow-collecting drives feed `speed_adaptive_verdict` and the Phase-5 roll-comp
verdict tool against the rewritten gates.
