# Lateral governor slew-bypass crossover plan

Status: proposed — plan only, not approval to change a live setting, Params key, or
deployment artifact
Date: 2026-07-17

## Scope

Determine whether removing the output governor's final slew stage (rate limiter and
sign-change unwind) — while retaining all production floor, caps, clipping, and
target-arrival blend — measurably improves ride comfort on a closed course without
regressing tracking, lag, torque-rate, saturation, or intervention rate.

**G0**: production `OutputGovernor` as checked in.

**G2**: `NoSlewGovernor` replay semantics — `_NoSlewHelperSet.approach()` always returns
`target`, and `previous_output` is zeroed before each active update. This bypasses the
final slew approach and sign-change unwind (both part of the slew stage, not caps).
Production floor, every restrict cap (`SAME_DIRECTION_LIMIT_CAP`, `SIGN_CONFLICT_CAP`,
`OVER_RESPONSE_MIN_SCALE`, `NEAR_ISO_ACCEL_CAP`, `OVERRIDE_RELEASE_CAP`), steering-rate
comfort blend, high-rate scaling, clipping, and target-arrival blend are retained. The
release backstop uses `approach()` and is therefore part of the bypassed stage.

**G1** (`same_direction_limit=False`) excluded — replay established G0 == G1 on the
metrics that matter.

## Non-goals

- No change to `StraightPathStabilizationMode` (remains `shadow` throughout; never Apply).
- No change to torque gain, friction floor, `LatControlTorqueV21`, or the response core.
- No change to hard platform safety limits in `card.py` or `opendbc`.
- No change to any Params key, Settings UI, or module outside
  `sunnypilot/custom/lateral/output_governor.py`.
- This test does not validate any other governor sub-feature (arrival blend, steering-rate
  comfort, over-response, ISO accel, sign conflict). Those are identical across conditions.
- Logs alone cannot justify a live change. Observational fixed-trace evidence is
  confounded and is not causal proof.

## Offline evidence (replay only, no causal claim)

Route `000002a0--9c3f2c80f6`, t=1835.8-1838.8:

| Metric | G0 | G1 | G2 |
|---|---|---|---|
| Output-minus-nominal RMS | 0.128940 | 0.128940 | 0.000475 |
| Old-direction torque p95 | 0.061 s | 0.061 s | 0 s |

G2 removes nearly all governor-imposed demand distortion in replay, establishing slew
limiting as the dominant source. It does not establish real-road comfort or safety.

A two-route model-free audit (`audit_governor_slew.py`) confirmed observational
fixed-trace evidence is confounded: placebo rank correlations 0.21-0.58 and held-out
angle coefficients flip sign. No live change is justified from logs.

## Prerequisites

This plan authorises **no code change** in any deploying branch. Before any on-track run,
the following must exist and be reviewed by a second engineer:

1. **Test-only implementation.** Separate proposal creates a reversible, non-shippable
   switch for G2 in `output_governor.py`, touching no Params key or deployment path.
   Must follow `param-settings-ui` workflow if it involves a tester-visible setting.
   Defaults to G0; never enabled in a commit merged to `master`.
2. **Replay parity.** Implementation reproduces B30 G2 metrics (RMS ~0.0005, p95 0 s).
3. **Fuzz invariants.** `fuzz_lateral_controller --open-loop --cases 500` passes.
4. **Analysis tool.** Reviewed tool (new or extended from `replay_output_governor.py`,
   `audit_governor_slew.py`, `lateral_comfort_imu.py`) segments rlogs by script phase,
   computes the metrics below, and produces a grouped comparison report. Tested on
   synthetic data before track day.

## Vehicle / track safety rules

Closed course or test track only. No public roads.

- **Safety driver**: belted, hands on wheel, monitoring the road, ready to override.
  Understands G2 permits larger torque steps.
- **Abort conditions**: any steering/brake/gas override, lateral-position departure
  > 0.5 m, torque-rate spike exceeding platform safety limit, unexpected oscillation,
  or communication/process alert. Aborted runs retaken at end of same band's block.
- **Restore on abort**: checkout returned to G0 before next run. Re-arm G2 only after
  confirming abort cause unrelated to condition.
- **Speed bands**: 12 m/s (city), 24 m/s (highway). Max 26 m/s.
- **No lead vehicle** (`radarState.leadOne.status == 0`).
- **Weather**: dry pavement, daylight, crosswind <= 5 m/s.

## Study design

### Randomization

Fixed block-randomized sequence before test day. If a reviewed, reversible
condition-selection method (coin-flip sequence, opaque-envelope draw, or equivalent) is
not available, the study is **BLOCKED** — do not proceed with non-randomized blocks.

Per band: 10 valid G0 + 10 valid G2 runs (20 per band, 40 total). Random permutation per
band, printed on a run sheet. No re-randomization mid-test. Driver and safety observer
are blind to condition; a third person reads the run sheet and confirms the correct
switch setting before each run.

Complete low-speed band first, then high-speed, nesting learning/fatigue within band.

### Valid repetition

Engaged from straight start through S-curve exit; no intervention; entry speed within
+/-0.5 m/s of target; no abort; complete telemetry.

## Script

### Straight segment

Rest at marked start, centered. Engage at 0 m/s, accelerate at ~1.5 m/s^2. Cruise >= 5 s
with no steering input. Continue directly into S-curve without disengaging.

### Gentle S-curve

Two consecutive constant-radius arcs in opposite directions. Radius >= 250 m for
high-speed band (target lateral accel <= 0.3 g at 24 m/s); same layout at lower speed
for low band. Mark arc entry and exit with cones. Straight recovery zone >= 50 m after
second arc. Disengage there, slow, return to start.

## Required telemetry

Every run produces one rlog with all signals below present and finite for the engaged
segment. Invalid if any required signal is missing or constant-false for > 0.5 s during
the S-curve.

**Governor state**: `controlsState.lateralControlState.torqueState.{output,desiredLateralAccel,actualLateralAccel,desiredLateralJerk,saturated,adaptiveTorqueState.{nominalOutput,governorReason,active,outputCap,steerLimitLimited,steerLimitSameDirection}}`

**Vehicle state**: `carState.{vEgo,aEgo,steeringAngleDeg,steeringRateDeg,steeringTorqueEps,steeringPressed,gasPressed,brakePressed}`

**Actuator**: `carControl.actuators.torque`, `carOutput.actuatorsOutput.torque`

**Body lateral accel**: `liveCalibration` (fed to `PoseCalibrator`), then calibrated
`livePose.angularVelocityDevice` z * `vEgo`.

**Lateral demand** (as available): `modelV2` / `controlsState.modelPathState`

**Lead verification**: `radarState.leadOne.status == 0` for all engaged frames.

**Condition marker**: A reason bit or new boolean on `torqueState` marks slew bypass
active. True for every G2 frame, false for every G0 frame. If a new cereal field is
needed, document in `docs/touch-points.md`.

## Route labeling

Label each run `{track}_{date}_{band}_{condition}_{run-number}`. Maintain a run-log CSV
with `label`, `condition`, `band`, `valid`, `abort_reason`, `entry_speed`,
`entry_lat_accel`, `notes`.

## Preregistered pass/fail thresholds

Fixed before the test. No post-hoc selection.

### Primary (comfort)

| Metric | Definition | Threshold |
|---|---|---|
| Body lateral-accel RMS (0.3-2 Hz) | Calibrated yaw-rate * vEgo, [0.3, 2] Hz Butterworth, over S-curve-steady segment | >= 20% reduction, G2 vs G0 median |
| Excess-event rate | Peaks exceeding 1.5x segment RMS per 100 s of S-curve-steady time | >= 20% reduction, G2 vs G0 median |

### Secondary (no-regression)

| Metric | Definition | Non-inferiority margin |
|---|---|---|
| Tracking-error p95 | p95 of |desired - actual lateral accel| over engaged segment | <= 5% increase |
| Curve-entry lag | Cross-correlation lag (desired to actual) in first 3 s of first arc | <= 20 ms increase |
| Torque-rate p95 | p95 of |d(torque)/dt| over engaged segment | No increase |
| Saturation fraction | Fraction of frames where |output| >= 0.95 * |max_output| | No increase |
| Intervention rate | Interventions per 100 engaged-km (pooled bands) | No increase |

### Stopping rule

After >= 6 valid runs per condition per band: if any primary metric shows G2 **worse**
than G0 with p < 0.05 (Mann-Whitney U, one-sided for wrong direction), stop NO-GO.

## Analysis

Per run: bandpass-filter body lateral accel, segment, compute metrics. Group by condition
and band. Report medians, percent change, Mann-Whitney U p-values, 95 % bootstrap CIs
(10 000 resamples). Secondary: one-sided non-inferiority at alpha = 0.05. Intervention
rate: Poisson exact 95 % CI on rate ratio.

Tool emits JSON with `verdict: "GO" | "NO-GO" | "INCONCLUSIVE"`.

## Decision table

| Primary pass? | Secondary pass? | Verdict |
|---|---|---|
| Yes (both bands) | Yes (all) | **GO** — see below |
| Yes (one band) | Yes (all) | **GO (conditional)** |
| Yes (any) | No (tracking or lag) | **NO-GO** |
| No | Yes or No | **NO-GO** |
| Stopping rule | - | **NO-GO (early stop)** |

A **GO** verdict does **not** authorise merging any bypass switch to `master`. It permits
a separate promotion proposal with its own route-corpus validation, track-to-public-road
generalisation analysis, safety case, and Param/workflow design. A GO alone means nothing
is merged.

**INCONCLUSIVE** (primary improves but secondary fails non-inferiority at p >= 0.05): may
recommend a follow-up with partial slew relaxation (e.g., doubled rate). A partial
relaxation is a new plan.

## Rollback (NO-GO)

1. Restore test checkout to `master` (production governor).
2. Delete the test-only switch and runner code.
3. File a brief ADR with failing thresholds and per-band metric table. If any secondary
   metric fails at p < 0.01, investigate whether a separate governor path needs correction.

## Plan status

This is a **plan only**. It authorises no code change in any branch, no Params key, no
deployment artifact, and no live behaviour change. The test may proceed only after every
prerequisite is met and reviewed.
