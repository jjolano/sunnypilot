# ADR: Global custom-recommended Resolution to custom-2.0

## Status

Accepted. Supersedes the rollout-gate portion of `2026-05-17-custom-longitudinal-v2-architecture.md`.

## Context

The initial custom-2.0 architecture kept `custom-recommended` unchanged until route replay, Drive Lab validation, and a road test promoted custom-2.0. The revised rollout chooses broader opt-in availability: users who explicitly select `custom-recommended` should receive the current custom-2.0 behavior globally, while unset or unknown `LongitudinalStack` values still resolve to `sunnypilot-current`.

## Decision

`custom-recommended` resolves globally to `custom-2.0`. This does not change the default baseline stack, does not let `sunnypilot-current` consume custom-only tuning, and does not remove stack-selector rollback to `sunnypilot-current`.

`custom-2.0` remains fail-closed: invalid custom output or internal custom-stack faults request immediate disable rather than silently falling back to baseline output.

## Consequences

- The global recommended alias trades per-platform promotion conservatism for a simpler opt-in custom rollout.
- Validation still uses manual-vs-custom route profiling and minimal baseline-vs-custom regression checks, but those checks are no longer a prerequisite for the alias to name custom-2.0.
- Telemetry must keep selected stack, selected personality, selected intent, selected reason, and rejected/suppressed reasons visible enough for rollback and route analysis.
