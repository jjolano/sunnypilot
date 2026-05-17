# ADR: Custom Longitudinal v2 Telemetry Compatibility

## Status

Accepted for initial implementation.

## Context

`custom-1.0` publishes stack selection and fallback shadow telemetry. `custom-2.0` is fail-closed and should not normally publish baseline shadow deltas because baseline output is not a runtime fallback path.

Cap'n Proto schema changes need a staged rollout so existing tools keep reading current fields while v2 becomes selectable.

## Decision

Stage 1 adds a stable `StackId.customV2` enum value and keeps the existing stack telemetry fields. `custom-2.0` publishes the selected stack and uses the existing fallback fields only for fault/latch state, not normal baseline-shadow comparison.

V2 policy telemetry will prefer stable text names for selected intent and rejected reasons before any later packed schema cleanup. Drive Lab support should initially parse the selected stack and intent/reason strings.

Stage 2 may remove `custom-1.0` and shadow fields after v2 is selectable and validated. Reusing old ordinals is acceptable only in that later cleanup because this fork controls the retained deployment path.

## Consequences

- Stage 1 is schema-compatible with existing stack telemetry readers.
- V2 fail-closed behavior is visible through an immediate-disable event and stack fallback reason text.
- Normal baseline accel delta telemetry is intentionally absent for v2.
