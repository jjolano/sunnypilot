# Uphill net-demand cap is shadow-only until calibrated

Status: accepted for shadow investigation and a calibration-gated Apply setting; on-road apply is not approved
Date: 2026-07-15
Relates to: [uphill net-demand acceleration-cap plan](../plans/2026-07-15-uphill-net-demand-accel-cap.md),
[clean-room longitudinal architecture](2026-06-13-clean-room-longitudinal-architecture.md), and
legacy [planner/MPC boundary](../legacy/adr/0001-longitudinal-planner-mpc-boundary.md).

## Context

The Toyota RAV4 TSS2 accepts an acceleration request through `ACC_CONTROL`; its PCM owns throttle
and gear selection. Avoiding a PCM downshift is therefore reachable only by shaping requested
acceleration, not by commanding a gear or throttle.

The proposed policy limits requested net demand on a sustained uphill:

```text
a_cmd <= ceiling - g * sin(relative_grade)
```

It is meant to defer discretionary speed gain once grade consumes most of the powertrain-demand
budget. It is not a maintain-speed-only or no-downshift-at-any-cost policy.

Current evidence does not justify actuation. On the steepest logged relative-grade bin,
openpilot's measured net-demand p95 is approximately `0.81 m/s²`, below the provisional driver
knee near `1.2 m/s²`. Routes 290 and 291 contain no sustained 6–10% climb that exercises the
proposal. Their raw pitch also carries roughly +0.04 rad offset, while coast-regression zero
estimates vary enough by route/window that raw pitch or a short session fit is not apply-grade
evidence.

Toyota's existing CarController has transition-only high-pass pitch compensation, a `long_pid`,
and accel windup/winddown limits. It has no explicit steady-grade feed-forward. The opendbc
submodule is not a customization point.

## Decision

1. Do not authorize or install an apply profile from the current corpus.
2. Permit a time-bounded shadow implementation in the custom longitudinal layer. Shadow computes
   the candidate after all finalizer arbitration and returns the existing target exactly. Expose
   an attested `Apply calibrated` setting now, but keep it pass-through unless a valid Drive
   Lab-produced steep-climb profile and the research-actuation gate are both present.
3. Estimate relative grade from calibrated `carControl.orientationNED[1]`, de-biased by a robust
   manual-coast regression. Use `livePose`/`liveCalibration` only for health, freshness,
   calibration identity, and source comparison in the first version.
4. Log the fit evidence, filtered grade, requested net demand, candidate cap, and would-be delta
   in an append-only `longitudinalPlanSP.longitudinalDebug` struct.
5. Keep the first possible apply tier leadless. A confirmed lead bypasses the cap so lead
   pullaway, gap recovery, standstill release, and catch-up cannot be starved.
6. Do not change MPC lead physics, the policy's existing coast estimator, `LongControl`, Toyota
   CarController, or any opendbc submodule file.
7. Add no guessed grade activation or hysteresis values. Effective apply requires labeled
   steep-climb data that identifies a repeatable shift/no-shift knee, estimator error and latency,
   and grade entry/exit thresholds that remain dormant on normal terrain.
8. If grade load alone exceeds the eventual ceiling, the cap may lower positive acceleration to
   zero but must not command uphill braking merely to preserve a gear.

An on-road apply promotion requires a follow-up ADR/status change after the plan's validation gates
pass. Until then, `apply` without a valid calibrated profile must sanitize to pass-through monitor
behavior, and the code default remains off.

## Rejected alternatives

- **Runtime RPM or numbered-gear plumbing:** RPM is deprecated in cereal, no car parser publishes
  it, and it is a downstream PCM outcome. Raw CAN RPM may label shifts offline only.
- **A hard-coded `1.2 m/s²` applied ceiling:** the value is a provisional driver knee on milder
  terrain, not a measured RAV4 shift threshold on the target climbs.
- **Raw pitch or a universal fixed offset:** the logs show mount/calibration bias and route-window
  fit movement.
- **Maintain-speed-only uphill logic:** stricter than the driver's demonstrated behavior and likely
  to make climbs feel gutless.
- **Steady-grade feed-forward or inverse Toyota compensation:** this proposal is an upper bound,
  not another controller; duplicating downstream pitch/PID behavior creates double compensation.
- **A parallel lead-aware model in the finalizer:** the first tier simply bypasses active leads
  and preserves the existing planner/MPC authority boundary.

## Consequences

- With no calibrated profile installed, the accepted stage has no driving effect and creates only
  bounded estimator/telemetry work.
- `finalize()` is the single clamp point, before `final_a_prev` is recorded and before
  the planner's ordinary accel clip.
- One hook-sized `plannerd` subscription change supplies optional source-health messages; those
  messages do not affect plan validity.
- A persisted grade profile is learned state and must be versioned, quality-gated, calibration-
  matched, and ignored on parse or evidence failure.
- The final Toyota request can differ transiently from the planner target; promotion margins are
  measured from `carOutput`/raw CAN rather than canceled with more custom control math.
- If a labeled steep corpus is not collected by 2026-08-15, the shadow runtime should be removed
  or parked rather than becoming permanent dormant debt.
