# Remove the shadow lateral disturbance classifier

Status: accepted
Date: 2026-07-16

## Context

The Phase 0b shadow disturbance classifier (`sunnypilot/custom/lateral/disturbance_classifier.py`)
classified every accepted torque-learning point (ACCEPT / QUARANTINE / REJECT_SHADOW) and
published counters on `liveTorqueParameters`, as a staging step toward gating learning samples
on classification. The program never graduated: the counters were never consulted in any
route diagnosis, the sufficiency-fixes plan (2026-07-02) explicitly declined to extend the
telemetry, and the learners shipped robust without it. The layers it staged for already handle
disturbances by design — upstream `torqued` filters the hard-reject conditions (lat inactive,
driver override, saturation) before points enter the learner, and the roll-comp/speed-aware
learners absorb transients statistically (`MIN_POINTS=2000`, span/sign gates, slope clipping,
blending, confidence floors).

## Decision

Delete the classifier, its live wiring, offline profilers, and tests without replacement:
the `torqued.py` shadow-classify hook (shrinking that upstream touch-point), the
`torqued_ext.py` counters, `lateral_disturbance_profile.py` / `profile_lateral_disturbances.py`,
and the classifier half of `fuzz_lateral_transitions.py` (the `LateralDemandPipeline` half
stays). The four capnp counter fields are renamed `*DEPRECATED` (ordinals retained).
Reintroduce sample gating only if learner params ever show spike contamination that the
statistical requirements fail to absorb.

## Consequences

- No runtime behavior changes: classification was shadow-only and never suppressed points.
- `fuzz_lateral_transitions.py` loses the `cooldown_hysteresis` scenario kind and
  classifier oracles; pipeline transition coverage is unchanged.
- Old rlogs still decode; the deprecated counter fields are simply no longer populated.
- `lateral_disturbance_sim.py` (physical stiction/backlash benchmarks) is unrelated and kept.
