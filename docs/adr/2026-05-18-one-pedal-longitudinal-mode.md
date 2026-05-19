# ADR: One-Pedal Longitudinal Mode Inside custom-2.0

## Status

Accepted.

## Context

`custom-2.0` normally treats cruise as a target and owns assertive progress policy. One-pedal behavior intentionally changes that mental model: driver lift-off should coast unless physical lead or stop evidence requires braking, while the cruise set speed acts as an acceleration ceiling. This is surprising enough to record because a future reader might otherwise re-enable speed-limit, curve, map, or excess-gap braking and acceleration in the name of normal custom-v2 behavior.

## Decision

One-Pedal Longitudinal is a default-off mode inside `custom-2.0`, exposed as `Off / Creep / Full Stop`; it is not a separate stack and never affects `sunnypilot-current`. Normal one-pedal operation suppresses non-hazard progress floors and no-hazard advisory braking, leaves Lead MPC and confident stop evidence authoritative for physical braking, and uses existing stack intent/reason strings for telemetry. Any cruise speed adjustment button enters Temporary Cruise Hold, restoring full normal custom-v2 cruise/advisory behavior until gas, brake, or disengagement returns to one-pedal.

## Consequences

- One-pedal does not change target follow gaps, Lead MPC danger gaps, FCW, or AEB behavior.
- Creep and Full Stop terminal behavior are one-pedal terminal policies, not Stop Approach evidence.
- No-hazard brake-light avoidance is best-effort at the planner boundary; platform controllers may still differ.
