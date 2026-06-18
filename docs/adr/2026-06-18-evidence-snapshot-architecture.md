# Evidence Snapshot architecture

Status: proposed
Date: 2026-06-18
Relates to: `CONTEXT.md`, `docs/legacy/CONTEXT-longitudinal.md`, `docs/legacy/CONTEXT-lateral-control.md`, and `docs/adr/2026-06-13-clean-room-longitudinal-architecture.md`.

## Context

Custom lateral and longitudinal features currently repeat raw-message extraction from `modelV2`, `radarState`, and `carState`. A tempting cleanup would be a new “world model” process or cereal schema, but that would add latency, source-freshness, schema, and communication-failure surface before there is evidence that a cross-process publisher is needed.

## Decision

Introduce **Evidence Snapshot** as a pure, stateless, in-process library under `sunnypilot/custom/evidence/`. It is a per-tick, non-authoritative summary of observed ego, model-path, lead, model-action, and source-health evidence. It uses frozen dataclasses, explicit source-health states/reasons, unit-suffixed fields, finite-or-`None` values, and no raw capnp/message references in the core snapshot.

Snapshots are constructed locally by each consumer from the messages it already has; there is no shared cache, singleton, service, or cross-loop timing dependency.

Snapshot construction is log-replayable: it may use only message fields and explicit source-status metadata available to runtime/replay callers, with no wall-clock reads, params reads, external state, or learned/persisted data.

Evidence Snapshot is mode-agnostic and contains no feature toggles, user personality, planner outputs, controller outputs, seed targets, target acceleration, processed curvature, or stop commitment. It may expose deterministic **Derived Evidence** such as same-tick lead risk or model stop distance, but longitudinal/lateral consumers retain ownership of scenes, candidates, intents, processed demand, trust learners, and final actuation targets.

## Consequences

- Phase 0 is contract and unit/parity tests only; no runtime behavior change.
- Initial implementation stays in custom code: `sunnypilot/custom/evidence/` plus custom lateral/longitudinal tests and wiring. It does not touch upstream `selfdrive/*`, `cereal/*`, or `services.py` in v0-v3.
- The first runtime consumer is lateral demand via a parity-tested adapter, because its seam is smaller and default-off/fail-closed.
- Custom longitudinal may consume Evidence Snapshot later, but only as evidence below the existing mode/custom-stack boundary.
- Quality-gated behavior waits until after extraction parity plus replay/fuzz validation. The first behavior use is disable-only guards: degraded or unknown source health may only reduce or withhold **Custom Authority**, never increase authority, relax caps, reduce MPC caution, or change stock behavior.
- Behavior validation must prove monotonic **Custom Authority**: degraded or unknown evidence leaves custom output unchanged or closer to stock/more conservative, with no higher curvature authority and no reduced lead/model caution.
- Evidence Snapshot gets no standalone user-facing param; it is an internal evidence boundary, and behavior changes remain guarded by the consuming feature's existing enablement.
- No new cereal schema, logging service, or `worldmodeld`-style process is introduced unless later route-analysis needs justify it.
