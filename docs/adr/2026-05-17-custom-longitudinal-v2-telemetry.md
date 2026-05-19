# ADR: Custom Longitudinal v2 Telemetry Compatibility

## Status

Accepted for initial implementation.

## Context

`custom-2.0` is fail-closed and does not publish baseline shadow deltas because baseline output is not a runtime fallback path.

Cap'n Proto schema cleanup can happen in this retained branch because this fork controls the deployment path and v2 is the only selectable custom stack.

## Decision

The stack schema publishes requested, resolved, and actuated stack IDs plus explicit v2 fault state. It does not include v1, baseline shadow stack, shadow accel, or runtime fallback fields.

V2 policy telemetry will prefer stable text names for selected intent and rejected reasons before any later packed schema cleanup. Drive Lab support should initially parse the selected stack and intent/reason strings.

## Consequences

- V2 fail-closed behavior is visible through an immediate-disable event and stack fault reason text.
- Normal baseline accel delta telemetry is intentionally absent for v2.
