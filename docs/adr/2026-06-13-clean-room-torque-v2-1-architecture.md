# Clean-room torque v2.1 architecture

Status: proposed
Date: 2026-06-13
Relates to: [restart plan Phase 2](../plans/2026-06-12-fork-restart-reimplementation.md),
legacy ADR `docs/legacy/adr/0002-torque-v2-1-refined-output-governor.md`

## Context

Torque v2.1 is the lateral controller the owner daily-drives and rates as the most
responsive. The restart reimplements it clean-room: best/cleanest architecture, behavior
preserved. This ADR fixes the architecture and — critically — the boundary between what is
safe to rewrite now and what is gated on validation data.

The old implementation is two very different bodies of code:

1. **Response core** (`latcontrol_torque_v2.py` ~lines 300–392, ~350 LOC): the math that
   produces the pre-shaping torque. Speed-interpolated KP schedule, 1 s lateral-accel
   request buffer with delay compensation, measurement smoother, low-speed same-sign
   unwind, low-demand friction scaling, the PID. **This is where the responsiveness lives.**

2. **Output stage** (~1200 LOC across four modules applied in series): over-response
   attenuator (28) → guarded response assist (440) → conservative output shaper (588) →
   refined output governor (v2.1-only, ~120). ~30 distinct behaviors, five stateful
   recovery-time trackers, and cross-module coupling (e.g. the controller passes
   `shaper_already_capped` into the governor's same-direction gate). This stack is the
   prime suspect for the unattributed v2.1 quirks.

These two bodies demand opposite treatment, which is the core of this decision.

## Decision

### 1. Module layout

All new code under `sunnypilot/custom/lateral/`:

```
sunnypilot/custom/lateral/
  response_core.py      # the ported math (pure, stateful only via small buffers)
  output_governor.py    # the unified replacement for the 4-module stack
  torque_v2_1.py        # LatControl subclass wiring core -> governor + diagnostics
  types.py              # frozen dataclasses for the inter-stage contracts
```

Integration is into upstream's **native** torque selector, not the fork's retired
registry: one entry in `latcontrol_torque_versions.json`, one dispatch case in
`sunnypilot/.../controlsd_ext.py:initialize_lateral_control()`. Both are listed in
`docs/touch-points.md`.

### 2. Response core — port unchanged-in-math, gate by exact parity (safe NOW)

The response core is reproduced with identical arithmetic and restructured for clarity
only. Because it is unchanged-in-math, it is validated **without route data**: a parity
test drives the new module and a flat transcription of the original formulas with the same
randomized input sequences and asserts equality to within fp tolerance. This catches
refactor/transcription errors, which is exactly the failure mode a structural rewrite
risks. The KP schedule, buffer length, smoother gains, unwind thresholds, and friction
scaling are consumed from `docs/legacy/tuned-constants.yaml`, not retyped.

Boundary: the response core owns everything up to and including the PID feedforward torque
and the signals the output stage consumes (`setpoint`, `measurement`, `desired_lateral_jerk`,
`actual_lateral_jerk`, `lookahead`-style terms). It does **not** own the NNLC/override
extension hook — that stays a separate, explicit injection point, not folded into the core.

### 3. Output stage — one unified governor, gated on the engaged-data corpus (NOT yet)

Replace the four series modules with **one** `OutputGovernor.update()` doing a single pass
over a single observation struct, in an explicit, documented precedence, emitting one
reason bitfield so every intervention is attributable (this attribution is the mechanism
that finally localizes the quirks).

The ~30 legacy behaviors collapse into **three conceptual operations** that are today
scattered across all four modules:

- **AUGMENT** — raise output to overcome under-response/actuator lag. Sources today:
  guarded-assist `assist_torque`/`bias_torque`, its curve-exit and curve-preposition
  boosts; the shaper's under-response catch-up bypasses and safety-limited under-response
  floor; the governor's under-response floor. Unify into one bounded, speed-scheduled
  augmentation term with one gate set (active, ¬steering_pressed, same-sign-hold,
  ¬low-demand, ¬bump, ¬saturated-learning-block).
- **RESTRICT** — cap output for safety/comfort. Sources today: the attenuator's
  over-response; the shaper's 15 caps (sign-conflict, over-response dynamic, near/over-ISO,
  bump, low-speed steer-limited, steering-rate comfort, actuator-lag comfort, stale-actuator
  reversal, safety-limited ramp/sign-hold, override release, same-sign unwind, high-speed
  actuator-lag unwind); the governor's high-rate and same-direction caps. Unify into one
  cap = min over applicable restrictions, expressed as a fraction of `max_output`.
- **RATE-LIMIT** — bound the rate of change. Sources today: the shaper's output rate limit
  with its recovery-rate schedule and five recovery-time trackers; the governor's slew,
  sign-change slew, and high-rate slew scaling. Unify into one slew with a binding rate =
  min over applicable rate sources, plus sign-change handling.

Application order per tick: `augmented = nominal + augment`; `capped = clip(augmented,
±cap·max_output)`; `output = slew(previous, capped, rate)` — with the augment term
permitted to relax cap and slew toward the unclipped value exactly as the legacy
under-response floor does (so catch-up is never defeated by the comfort caps). The
behavioral catalog above is the migration checklist: every legacy behavior is mapped to
KEEP (folded into one of the three operations), MERGE (several legacy reasons → one), or
DROP-as-redundant, with a one-line justification recorded in `output_governor.py`.

This is a behavior-changing rewrite. It is therefore **gated on the engaged-route corpus**
(Phase 0): feel-parity is measured by replaying engaged routes through old v2.1 and the new
controller and diffing tracking error, oscillation score, and per-reason intervention
rates. As of 2026-06-13 the corpus is ~all manual driving (~2.4 min engaged) — so this
gate cannot yet be met. Property tests (output continuity under input steps, cap
monotonicity, augment≤floor≤unclipped ordering, sign-change boundedness, reset behavior)
gate the governor's *invariants* without route data, but they do **not** certify feel.
Implementation of the governor proceeds to a property-tested first cut; promotion to the
default `TorqueControlTune=2.1` waits for engaged-data parity.

### 4. Diagnostics

One compact reason bitfield per operation (augment/cap/slew), logged so a single field
attributes any intervention. This replaces the legacy `adaptiveTorqueState` sprawl. Cereal
additions are deferred until the governor's field set stabilizes; the first cut logs via a
minimal struct, not the legacy 114-line telemetry block.

## Consequences

- The response core can be built and proven now; the governor can be built now but only
  property-gated now — its feel-parity sign-off is blocked on engaged data, making engaged
  drives on the current build (before the restart deploys) a prerequisite to finishing
  Phase 2. This is the single most schedule-relevant consequence.
- Collapsing four modules into three operations with one precedence removes the cross-module
  coupling (`shaper_already_capped` and friends) that is the leading quirk hypothesis; if a
  quirk survives the merge, the reason bitfield says which operation produced it.
- Risk: a subtle behavior (e.g. stale-actuator reversal, low-speed corrective-unwind
  catch-up, safety-limited sign-hold) is dropped as "redundant" when it was load-bearing.
  Mitigation: the catalog forces an explicit KEEP/MERGE/DROP verdict per behavior, and the
  engaged-replay gate is required to catch a wrong DROP before default promotion.
- We inherit upstream's torque-selector UX and v0/v1 fallbacks for free; the fork keeps no
  parallel torque registry.
