# Uphill net-demand acceleration-cap plan

Status: staged implementation complete; no calibrated apply profile exists; actuation is not approved
Date: 2026-07-15
Decision record: [Uphill net-demand cap is shadow-only until calibrated](../adr/2026-07-15-uphill-net-demand-cap.md)

## Outcome

The proposed finalizer clamp is mechanically sound, but the feature is not calibrated enough to
actuate. Routes 290 and 291 confirm that openpilot's engaged uphill demand remains below the
provisional `1.2 m/s^2` knee; they do not contain the sustained 6–10% climbs on which the cap is
supposed to earn its keep. The grade zero, grade activation threshold, filter latency, and true
Toyota PCM downshift knee therefore remain unproven.

Proceed with time-bounded shadow collection. The Settings UI exposes `Apply calibrated`, but the
runtime must reduce it to pass-through/monitor behavior unless a versioned steep-climb profile and
the existing research-actuation gate are both valid. No guessed threshold may gain authority.

## Verified architecture

### Command path and exact clamp point

The command path in this checkout is:

1. `selfdrive/controls/lib/longitudinal_planner.py` obtains the MPC and model acceleration targets.
2. `LongitudinalPlannerSP.final_longitudinal_output()` calls
   `CustomLongitudinalFinalizer.finalize()`.
3. `finalize()` calls `_finalize_impl()`. Its result has already passed stop-hold/release,
   SCC stop/curve/cut-in caps, launch-dip damping, the follow coast band, lead-catch-up capping,
   and approach damping.
4. The planner clips the returned value to the ordinary accel/turn envelope and publishes it as
   `longitudinalPlan.aTarget`.
5. `controlsd` passes `longitudinalPlan.aTarget` to `LongControl.update()`. The PID plus
   feed-forward output becomes `carControl.actuators.accel`.
6. `card.py` passes `carControl` to the Toyota interface. Toyota's controller rate-limits the
   accel request, applies its transition-only pitch compensation and `long_pid`, and sends
   `ACC_CONTROL`. The PCM owns throttle and gear selection.

The one complete post-arbitration choke point is the public `finalize()` wrapper, immediately
after `_finalize_impl()` and before `final_a_prev` is recorded. Add one final stage there. This
covers the disabled, E2E, stop-hold, and SCC/ACC return paths without duplicating a clamp at every
return. `final_a_prev` must store the post-cap value so stop-release and launch damping continue to
observe the actual planner command.

`finalATargetUnclipped` will then mean “after the net-demand stage, before the ordinary planner
accel clip.” The new trace must retain the pre-cap value so shadow/apply attribution is not lost.

### Grade-signal path

In this checkout, `controlsd.py`, not `card.py`, reads `livePose`, applies `PoseCalibrator` using
`liveCalibration`, and publishes the calibrated `[roll, pitch, yaw]` list as
`carControl.orientationNED`. `card.py` consumes the resulting `carControl` but does not populate
the orientation.

Use calibrated `carControl.orientationNED[1]` as the grade value. Add optional `livePose` and
`liveCalibration` subscriptions to `plannerd` only for source validity, standard deviation,
freshness, calibration fingerprinting, and a shadow comparison with raw device-frame pitch. Do
not add either service to `PLANNER_VALIDITY_CHECKS`; loss of grade evidence must disable only this
cap, not invalidate `longitudinalPlan`.

Direct `livePose.orientationNED.y` is not the apply source in the first version. It is lower-level
and has useful health fields, but it is in the device frame and would duplicate the existing
calibration transform. Shadow telemetry will show whether it has a material latency/noise
advantage before that choice is revisited.

### Cheap verification of the existing evidence

The cached arrays were analyzed with `uv run --extra testing --extra tools python` and numpy only.

| Evidence | Route 290 | Route 291 | Meaning |
|---|---:|---:|---|
| Raw `carControl` pitch median | 0.0383 rad | 0.0423 rad | Confirms the roughly +0.04 rad offset; raw zero is unusable. |
| Exact manual-coast OLS slope | -7.02 | -7.94 | Correct negative sign, but not a universal fixed `-g` fit. |
| Exact manual-coast `pitch0=-b/a` | 0.0142 rad | 0.0038 rad | A per-route/session fit moves enough to require quality and stability gates. |
| Engaged steep-bin net-demand p95 | approximately 0.81 m/s² pooled | approximately 0.81 m/s² pooled | Confirms the established dormant-cap caveat. |

Five- and ten-minute rolling coast fits were more stable than 30–120 second fits, but their
route/window spread was still material. A short boot-time regression is therefore not sufficient
authority for an applied cap. The current routes remain useful as the normal-terrain no-regression
corpus, not as the steep-grade calibration corpus.

## Runtime design

### One module and one final stage

Add `sunnypilot/custom/longitudinal/net_demand_cap.py`. Keep the estimator, cap calculation, and
their typed results together; a second abstraction is unnecessary until another consumer exists.

The custom adapter owns the estimator because it already reads Params and extracts `carState`,
`carControl`, and other source health. It places a typed grade estimate and sanitized cap config on
`CustomLongitudinalOutput`. The finalizer's new last stage consumes that estimate and the
post-arbitration `a_target`, then returns a typed trace through `_TelemetryAdapter.result`.

Do not alter MPC inputs, policy candidates, `LongControl`, Toyota code, or the existing
`DragEstimator`/coast policy in this change.

### Relative-grade estimator

Use the requested coast-regression contract:

```text
aEgo = slope * pitch + intercept
pitch0 = -intercept / slope       (accepted only when slope < 0)
relative_pitch = pitch - pitch0
grade_accel = g * sin(relative_pitch)
```

The online estimator has two deliberately different time scales:

1. **Slow zero estimation.** Collect only source-healthy, moving, manual coast samples:
   `not carControl.longActive`, gas off, brake off, not standstill, finite `vEgo/aEgo/pitch`.
   Treat engaged frames as on-throttle even when the driver's gas flag is false; the PCM can be
   applying power. Decimate correlated 20 Hz samples, retain a bounded rolling window, and run
   the exact route-wide two-term fit. Reject residual outliers with a median/MAD pass and recompute
   on a slow cadence. Run the same fit within bounded speed bands as a diagnostic so
   speed-dependent drag cannot silently move the intercept; disagreement blocks offline profile
   promotion but does not redefine the requested `pitch0=-b/a` estimate.
2. **Fast grade tracking.** Subtract the last accepted `pitch0`, then use a short causal median
   prefilter followed by a first-order low-pass. Start shadow evaluation with a 0.25 s median
   window and a 0.35 s time constant; the Drive Lab sweep must either validate or replace these
   before apply. These are shadow candidates, not promoted tuning constants.

An accepted zero/profile requires all of the following, with numeric thresholds selected from the
shadow corpus rather than guessed into actuation:

- negative slope with a physically plausible magnitude;
- enough independent coast samples and eligible coast time;
- enough pitch span to identify a slope rather than fit noise;
- bounded residual MAD and useful fit score;
- a stable `pitch0` across consecutive fit windows and repeated routes;
- a matching live-calibration fingerprint and valid schema version.

Install only a Drive Lab-approved profile as versioned JSON in an internal Param. Store `pitch0`,
slope, sample/eligible-time count, pitch span, residual MAD, fit score, and the calibration
fingerprint. A missing, corrupt, stale, incompatible, or calibration-mismatched profile is “not
ready,” never a default zero. Adopt a revised zero only while the cap is
ineligible/longitudinal control is inactive;
never retare the estimator while an applied cap is binding. The on-device coast fit remains shadow
evidence and never promotes itself into an apply profile.

Vehicle pitch can move during accel/jerk even when road grade does not. Mark such samples dynamic,
withhold them from zero fitting, and hold the last trusted fast-grade estimate only for a bounded
age. The shadow analyzer must choose the accel/jerk gate and maximum hold age. If that age expires,
the cap becomes ineligible; do not let accel-induced nose-up create a positive-feedback cap.

### Source quality, latency, and degraded behavior

Each tick validates:

- `livePose.orientationNED.valid`, `inputsOK`, `sensorsOK`, and `posenetOK`;
- finite/bounded `yStd` and a fresh `livePose` receive age;
- calibrated `liveCalibration` with the profile's calibration fingerprint;
- a finite three-element `carControl.orientationNED`;
- a ready/stable coast-regression profile;
- a fresh filtered estimate rather than a held stale value.

The source-age/std and dynamic-motion bounds are outputs of the shadow analysis. They are not
apply defaults until the corpus shows their healthy distributions.

Expected missing/invalid evidence is degraded operation: clear the cap regime, return the
uncapped target, and log a stable block reason. An internal exception or non-finite cap output in
shadow disables only shadow computation. In the calibration-gated apply mode, an internal fault after the cap
has bound during the engagement uses the existing `customLongitudinalFault` latch and immediate
disable path; ordinary evidence loss remains a non-faulting pass-through.

### Cap math

For a healthy uphill estimate and a positive post-arbitration command:

```text
grade_accel = g * sin(filtered_relative_pitch)
requested_net_demand = a_target_before + grade_accel
candidate_a_cap = max(0.0, ceiling - grade_accel)
a_target_after = min(a_target_before, candidate_a_cap)
```

The `max(0, ...)` floor is intentional. If grade load alone exceeds the ceiling, avoiding a shift
would require commanded deceleration and accepted speed sag. This feature defers discretionary
speed gain; it does not brake uphill to preserve a gear. Log this case separately as
`grade_load_exceeds_ceiling` because a downshift may be unavoidable even at zero desired accel.

Shadow mode always returns `a_target_before` exactly. It still logs `candidate_a_cap`,
`requested_net_demand`, and the would-be delta.

### Grade activation and hysteresis

Calibration-gated Apply uses a separate two-state `HOLD`/`CAP` latch following the pattern of
the finalizer's existing follow-band `HOLD`/`DECEL` state:

- enter `CAP` only after filtered percent grade is at or above `grade_enter_percent` for the
  calibrated entry dwell;
- remain in `CAP` through the dead band;
- exit to `HOLD` only after grade is at or below
  `grade_enter_percent - grade_hysteresis_percent` for the calibrated exit dwell;
- reset to `HOLD` immediately on invalid/stale evidence, pedals, inactive longitudinal control,
  stop/release context, or a lead bypass.

Shadow must log valid positive-grade samples without a runtime activation threshold so the
analyzer can sweep enter, exit, and dwell values. Do not commit a guessed threshold. The Apply UI
option is exposed now, but the runtime receives enter/exit/dwell values only from a calibrated
profile; without one, Apply is exact pass-through monitor behavior.

### Lead-following interaction

The first apply tier is leadless only. Any active `radarState` lead bypasses the cap while the final
target is positive. Shadow still calculates and labels the lead-present candidate for analysis,
but actuation does not risk starving lead pullaway, standstill release, gap recovery, or moving
lead catch-up.

This broad bypass is intentionally smaller than inventing another lead-risk model in the
finalizer. If later data justifies a lead-aware tier, it must reuse the existing lead context and
`lead_catchup_accel_cap` geometry; it is a separate decision.

Also bypass driver gas/brake, force-decel, `should_stop`, stop-hold/release, nonpositive targets,
and speeds outside the calibrated steep-climb envelope. A lower existing braking/advisory target
always wins because the cap is a `min` operation.

### No double compensation

The cap subtracts estimated road load once from an upper bound; it does not add grade
feed-forward. Leave Toyota's high-pass `pitch_compensation`, `long_pid`, accel windup/winddown,
and downhill-only `PERMIT_BRAKING` gate untouched in the opendbc submodule.

Toyota's high-pass term can temporarily change the final CAN accel request after the planner cap.
Calibrate that downstream response from `carOutput.actuatorsOutput.accel`/raw `ACC_CONTROL`, and
leave enough measured margin below the shift knee. Do not add an inverse Toyota compensation in
the custom layer; that would duplicate a downstream transition controller and still be wrong on
steady grade.

## Params and Settings UI

Staged implementation:

| Param | Type/default | UI/behavior |
|---|---|---|
| `UphillNetDemandCapMode` | `STRING`, `off`, `PERSISTENT + BACKUP` | Cruise page: `Off` / `Monitor only` / attested `Apply calibrated`. Unknown values fail closed to `off`; Apply is effective only with a valid profile and research gate, otherwise it is pass-through monitor behavior. |
| `UphillNetDemandCeiling` | `FLOAT`, `1.2`, `PERSISTENT + BACKUP` | Cruise-page numeric control in m/s². In shadow it is an analysis reference and makes no driving change. Runtime rejects non-finite/out-of-range values. |
| `UphillNetDemandGradeProfile` | versioned `STRING` JSON, empty, `PERSISTENT + DONT_LOG` | Internal Drive Lab calibration artifact; no Settings UI control. Missing/invalid means estimator not ready. |

Reuse `LongitudinalDebugTraceMode=log` to publish the detailed trace and
`AllowLongitudinalResearchActuation` as the apply gate. The collection deployment sets the
new mode to `shadow` and debug trace to `log`; code defaults remain off.

The internal calibrated profile, not extra user-facing tuning Params, carries grade enter/exit,
dwell, filter, source-quality, speed-envelope, and calibration-match values. Apply remains gated by
Custom Longitudinal, openpilot longitudinal control, and the existing research-actuation toggle.
The Settings copy explicitly says it can defer uphill speed gain and is pass-through without a
valid profile. Regenerate `sunnypilot/sunnylink/settings_ui.json` and cover mode/numeric encodings.

## Log contract

Append `LongitudinalDebug.uphillNetDemandCap @26 :UphillNetDemandCapTrace` in
`cereal/custom.capnp`; never reuse the retired scenario-context slot. Attach it to the typed
telemetry snapshot returned by `_TelemetryAdapter.result`, then publish it through the planner trace path only when
`LongitudinalDebugTraceMode=log`.

The trace fields are:

| Group | Fields |
|---|---|
| Mode/verdict | `mode`, `effectiveMode`, `eligible`, `wouldCap`, `applied`, `blockReason`, `regime` |
| Source/quality | `source`, `sourceAgeS`, `carPitch`, `livePosePitch`, `pitchZero`, `relativePitch`, `filteredGradePercent`, `profileReady` |
| Fit evidence | `fitSlope`, `fitScore`, `fitPitchSpan`, `fitResidualMad`, `fitSampleCount`, `fitSpeedBandSpread` |
| Cap evidence | `ceiling`, `gradeEnterPercent`, `gradeExitPercent`, `gradeAccel`, `aTargetBefore`, `aTargetCap`, `aTargetAfter`, `requestedNetDemand`, `deltaA`, `gradeLoadExceedsCeiling` |
| Gating evidence | `gradeHeld`, `researchActuationAllowed`, `hasLead` |

`applied` must always be false and `aTargetAfter == aTargetBefore` in shadow or in Apply without a
valid profile/research gate.
Existing `finalATargetUnclipped`, `finalATargetClipped`, `accelClipMin`, and `accelClipMax` remain
the authoritative downstream values.

## Drive Lab calibration

### Extract once

Extend `.agents/skills/route-drive-diagnosis/scripts/extract_route_npz.py` so one rlog pass adds:

- `carControl.orientationNED[1]`, enabled/long-active, and requested accel;
- `carOutput.actuatorsOutput.accel`;
- raw `livePose` pitch/valid/std/health and `liveCalibration` status/pitch;
- the complete `uphillNetDemandCap` trace;
- lead status/distance/relative speed for bypass validation;
- raw Toyota CAN address 452 payloads as an optional offline RPM outcome label.

The product never consumes RPM. Offline RPM/speed-ratio change points may label a PCM downshift
after the fact because they are useful outcome evidence, not control authority. If address 452 is
not reliable on this RAV4, require driver bookmarks plus a repeatable accel/jerk signature; do not
promote on guessed shift labels.

Add `tools/drive_lab/profile_uphill_net_demand.py`. It reads only the extracted NPZ and:

1. reproduces the manual-coast regression and reports slope/zero/span/residual/window stability;
2. compares car-frame pitch, raw live-pose pitch, and online filtered grade, including crossing
   latency and false activation counts;
3. reports measured net demand (`aEgo + grade_accel`), planner-requested net demand
   (`aTargetBefore + grade_accel`), and final Toyota-request net demand
   (`carOutput.accel + grade_accel`);
4. stratifies manual vs engaged, leadless vs lead-present, speed, grade, and mode;
5. joins downshift labels and brackets the knee with the highest repeated no-shift demand and the
   lowest repeated shift demand;
6. sweeps filter, enter-grade, hysteresis, and dwell choices without rereading logs;
7. emits a JSON verdict/config recommendation and normal-route invariant report.

Choose the ceiling only if the shift/no-shift brackets separate. Set planner-domain margin from
the observed p95 estimator error plus the p95 difference between finalizer target and Toyota's
final accel request. If the brackets overlap materially, there is no dependable scalar knee and
the feature remains NO-GO.

Choose the grade-entry threshold as the lowest sustained grade where labeled shifts are avoided
by the candidate while routes 290/291 stay inactive. Choose hysteresis/dwell from measured
filtered-grade noise so a sustained climb has one entry and one exit, not repeated state changes.

### Steep-climb capture

Collect at least three repeat passes over the same safe sustained 6–10% climb, long enough to
include settled grade before and after the crest:

- manual baseline passes with gradual demand increases and driver bookmarks for every felt/heard
  PCM shift;
- shadow-engaged leadless passes at matched speeds, including a modest cruise-speed gain request;
- one shadow lead-follow pass to prove the bypass/catch-up telemetry;
- at least two speed bands representative of real use;
- route metadata for payload, HVAC, weather, direction, and unusual traffic.

Prefer a paired manual then engaged pass so the same route supplies coast-fit and automated
evidence. Do not intentionally force an unsafe maneuver to create a shift label.

## Validation gates

### Shadow gate

- Bit-for-bit equality of finalizer output in `off` vs `shadow` for deterministic fixtures and
  replay: `aTargetAfter == aTargetBefore`, `applied == false` on every frame.
- Unknown modes and malformed numeric/profile Params fail to off/not-ready without blocking other
  custom-source refresh.
- Estimator synthetic tests recover a known pitch zero with outliers, reject wrong-sign/low-span
  fits, invalidate calibration mismatch, and never emit non-finite values.
- Routes 290/291 preserve existing planner/actuator outputs and reproduce the established
  approximately `0.81 m/s²` engaged steep-bin p95.

### Estimator/calibration gate

- A stable zero across at least three paired steep-route passes; cross-route spread and residual
  error must be small enough that their p95 load error fits inside the measured downshift margin.
- Causal grade crossing latency meets the measured requirement (target: p95 no worse than 1.0 s)
  without normal-road false entries or climb chatter.
- Source invalid/stale/dynamic windows reliably become ineligible and never retain a stale cap.
- Car-frame pitch is chosen over direct live-pose pitch unless the comparison demonstrates a
  material, repeatable advantage from the raw source.

### Apply-promotion gate

- At least three labeled downshifts and three matched no-shift high-demand windows establish a
  non-overlapping or usefully separated knee.
- The proposed cap fires only when the grade latch is active and requested net demand would cross
  the calibrated knee; it never raises accel or weakens a negative target.
- On controlled shadow replay, the simulated final Toyota request stays below the knee including
  downstream pitch/PID transients.
- Lead-present frames have zero applied-cap count in the first tier; launch, release, stop, pedal,
  and force-decel fixtures are exact pass-through.
- Routes 290/291 and a fresh normal-terrain route have zero applied-cap frames and unchanged
  longitudinal metrics.
- A guarded on-road A/B then confirms fewer/no provoked shifts without unacceptable speed sag,
  catch-up delay, accel chatter, or driver intervention.

If any gate fails, keep shadow only long enough to diagnose. If no labeled steep corpus exists by
2026-08-15, remove/park the runtime shadow feature rather than carrying permanent telemetry debt.

## Files and tests

### New files

| File | Purpose |
|---|---|
| `sunnypilot/custom/longitudinal/net_demand_cap.py` | Coast-fit grade estimator, typed result/config, shadow/apply cap state. |
| `sunnypilot/custom/longitudinal/tests/test_net_demand_cap.py` | Estimator, cap invariant, hysteresis, source failure, lead bypass, and shadow equality tests. |
| `tools/drive_lab/profile_uphill_net_demand.py` | NPZ-only calibration/verdict analyzer. |
| `tools/drive_lab/tests/test_profile_uphill_net_demand.py` | Synthetic fit and conservative NO-GO report tests; knee/threshold promotion tests wait for labeled data. |

The staged runtime, trace, Settings controls, extractor, and conservative Drive Lab profiler are
implemented. No calibrated profile is checked in or installed.

### Modified files in the staged implementation

- `sunnypilot/custom/longitudinal/wiring.py` — sanitize Params, update estimator, carry typed
  grade/config evidence; isolate failed reads from existing source toggles.
- `sunnypilot/custom/longitudinal/finalizer.py` — one post-`_finalize_impl` stage and typed result
  telemetry before recording `final_a_prev`.
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py` — retain/publish the new finalizer
  trace; no arbitration duplication.
- `selfdrive/controls/plannerd.py` — subscribe to optional `livePose`/`liveCalibration`, without
  making plan validity depend on them.
- `cereal/custom.capnp` — append `UphillNetDemandCapTrace` at ordinal 26.
- `common/params_keys.h` — off/shadow/apply mode, ceiling, and internal calibrated-profile keys.
- `sunnypilot/sunnylink/settings_ui_src/pages/cruise.yaml` and generated
  `sunnypilot/sunnylink/settings_ui.json` — Off/Monitor/Apply and ceiling controls with explicit
  calibrated-only/pass-through copy.
- `sunnypilot/sunnylink/tests/test_compile_settings_ui.py`,
  `sunnypilot/selfdrive/ui/settings_schema/tests/test_cruise_panel.py`, and
  `sunnypilot/selfdrive/ui/settings_schema/tests/test_encoding.py` — schema and encoding coverage.
- `sunnypilot/custom/longitudinal/tests/test_wiring.py` and
  `test_longitudinal_debug_trace.py` — Param refresh isolation and append-only telemetry coverage.
- `.agents/skills/route-drive-diagnosis/scripts/extract_route_npz.py` — one-pass extraction of
  grade/cap/downstream command and optional offline shift labels.
- `docs/touch-points.md` — update the existing rows below; do not add a submodule row.

### `docs/touch-points.md` row updates

- `common/params_keys.h`: add the off/shadow/apply cap, ceiling, and internal profile keys.
- `selfdrive/controls/plannerd.py`: add optional `livePose`/`liveCalibration` subscriptions for
  grade health/calibration; no output-validity dependency.
- `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`: add finalizer cap telemetry
  publication only; arbitration stays in the custom finalizer.
- `cereal/custom.capnp`: append the structured uphill net-demand shadow trace.

No change is planned in `opendbc_repo`, `selfdrive/controls/controlsd.py`, `LongControl`, Toyota
CarController, or any gear/RPM product schema.

### Verification commands for the staged shadow/apply change

```bash
uv run ruff check \
  sunnypilot/custom/longitudinal/net_demand_cap.py \
  sunnypilot/custom/longitudinal/finalizer.py \
  sunnypilot/custom/longitudinal/wiring.py \
  tools/drive_lab/profile_uphill_net_demand.py

uv run --extra testing --extra tools python -m pytest \
  sunnypilot/custom/longitudinal/tests/test_net_demand_cap.py \
  sunnypilot/custom/longitudinal/tests/test_wiring.py \
  sunnypilot/custom/longitudinal/tests/test_finalizer_characterization.py \
  sunnypilot/custom/longitudinal/tests/test_longitudinal_debug_trace.py \
  sunnypilot/custom/longitudinal/tests/test_shadow_observability.py \
  tools/drive_lab/tests/test_profile_uphill_net_demand.py \
  sunnypilot/sunnylink/tests/test_compile_settings_ui.py \
  sunnypilot/selfdrive/ui/settings_schema/tests/test_cruise_panel.py \
  sunnypilot/selfdrive/ui/settings_schema/tests/test_driving_panel.py \
  sunnypilot/selfdrive/ui/settings_schema/tests/test_encoding.py

uv run --extra testing --extra tools python -m openpilot.tools.drive_lab.fuzz_longitudinal --preset openpilot-acc --cases 100
uv run --extra testing --extra tools python -m openpilot.tools.drive_lab.fuzz_longitudinal --preset ncap-acc --cases 100
git diff --check
```

## Recommendation

**GO for shadow-only instrumentation and steep-climb data capture. NO-GO for an applied uphill
net-demand cap on the current evidence.**
