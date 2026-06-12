# Scene-Memory Planner Stack Design

## Summary

Introduce a top-level **Planner Stack** selector and the first non-default planner family, `scene-memory-v1`. The long-term destination is a full longitudinal planner rewrite that uses volatile short-horizon **Scene Memory** for source confidence, freshness, eligibility, and continuity across all longitudinal sources that the active **Longitudinal Mode** allows.

The first implementation milestone is not the full behavior rewrite. It creates the durable boundary, UI, telemetry, validation gate, and legacy-equivalent shadow-memory scaffold needed to make later active behavior testable and reversible without duplicating existing source classification.

## Locked decisions

- Add a top-level `PlannerStack` concept now rather than encoding planner families inside `LongitudinalStack`.
- Initial values:
  - `planner-current`: default baseline planner family, preserving current planner behavior.
  - `scene-memory-v1`: first scene-memory planner family.
- `PlannerStack` resolution is independent from `LongitudinalStack` and controls-profile selection.
- `PlannerStack` selection is latched for an onroad cycle; no hot-switching while active.
- `scene-memory-v1` is the destination for a full longitudinal planner rewrite, but milestones must be staged.
- Scene memory is volatile and short-horizon only; it resets across process/onroad-cycle boundaries.
- No persistent SLAM map, route obstacle memory, or permanent lane/world truth belongs in `scene-memory-v1`.
- In Milestone 1, Scene Memory consumes existing scene-like artifacts, including lead context, SCC evidence, planner candidates, speed/map state, and current planner output; it does not reclassify sources or mutate output.
- Later active Scene Memory behavior must respect **Longitudinal Mode** source boundaries.
- Later active Scene Memory behavior must not override **Longitudinal MPC** physical feasibility.
- After authority begins in a later active milestone, planner-stack faults are **Fail-closed**: request disable rather than silently actuating `planner-current`.
- Pre-engagement **Compatibility Fallback** may resolve an unavailable requested planner stack to `planner-current`.
- The normal cruise settings UI may expose a **Planner Stack Setting**, but active `scene-memory-v1` selection is blocked until the **Planner Validation Gate** passes.
- Planner-stack and scene-memory telemetry get dedicated `longitudinalPlanSP` fields; do not overload existing `stack`, `longitudinalMode`, or `decisionLayer` meanings.

## Non-goals

- No persistent world model.
- No lateral authority in the first milestone.
- No per-tick selector arbitration between planner families.
- No runtime fail-open fallback from invalid active scene-memory output to baseline output.
- No bypass of ACC/E2E/SCC source boundaries.
- No replacement of MPC lead/cruise physics in the first scene-memory architecture.
- No duplicate lead/SCC/speed/map classification in Milestone 1.

## Architecture

```text
model/radar/map/carState/controls state
        ↓
Longitudinal Mode source eligibility
        ↓
existing classifiers/resolvers/candidates
  - PrimaryLeadContext
  - SccEvidenceResult/SccEvidenceSelector
  - LongitudinalCandidate/Decision telemetry
  - speed-limit/map/SCC provider state
        ↓
PlannerStack resolver
        ↓
planner-current OR scene-memory-v1
        ↓
Scene Memory snapshot inside scene-memory-v1
  - source freshness
  - confidence and stability
  - source eligibility
  - recent lead/path/map/speed/stop evidence
  - validation and fault status
        ↓
Longitudinal MPC physical feasibility where applicable
        ↓
longitudinalPlan + longitudinalPlanSP telemetry
        ↓
existing controllers
```

The selector chooses a planner family at the onroad-cycle boundary. It is not a behavior planner and must not choose between candidates every tick.

In Milestone 1, `scene-memory-v1` is a shadow snapshot over the existing planner artifacts. It must not become a parallel lead classifier, SCC classifier, speed-limit resolver, map resolver, or MPC replacement.

## Source authority boundary

`scene-memory-v1` may eventually reason across all longitudinal source families:

- cruise/driver intent
- lead follow and lead transitions
- model stop/slowdown evidence when mode-eligible
- SCC curve vision when mode-eligible
- SCC map/advisory curve when mode-eligible
- speed-limit assist when mode-eligible
- OSM/map caution when mode-eligible
- launch, pullaway, and stop/go state

However, source eligibility remains owned by **Longitudinal Mode**. For example, ACC must not consume model-stop, map, speed-limit, OSM, or SCC curve actuation candidates just because `scene-memory-v1` remembers them.

Milestone 1 records source eligibility and freshness only. It does not grant new source authority.

## Runtime fault policy

Before authority begins:

- If requested `PlannerStack` is unavailable, resolve to `planner-current` with a compatibility reason.
- UI should show the requested and resolved values clearly.

After scene-memory authority begins in a later active milestone:

- Invalid scene-memory output requests disable.
- Non-finite values, stale required output, broken trajectory shape, source-boundary violations, or impossible state transitions are faults.
- Low confidence should degrade inside the planner by restricting or holding behavior; it should not silently switch planner families.

## UI policy

Add a normal cruise-settings **Planner Stack Setting** that displays `planner-current` and the validation-gated status of `scene-memory-v1`.

Until the **Planner Validation Gate** passes:

- `planner-current` remains the only active selectable value.
- `scene-memory-v1` may be shown as unavailable/validation-gated or omitted from the selection list.
- Direct param forcing of an unvalidated `scene-memory-v1` must resolve to `planner-current` before engagement.
- Direct param forcing of a validated `scene-memory-v1` during Milestone 1 may resolve for telemetry, but actuation remains `planner-current` until a later active implementation lands.

## Telemetry

Add dedicated `longitudinalPlanSP` telemetry for planner-stack and scene-memory status.

Minimum planner-stack fields:

- requested planner stack
- resolved planner stack
- actuated planner stack
- compatibility fallback reason
- validation-gate status
- fail-closed/fault reason

Minimum scene-memory fields:

- enabled/active/shadow state
- memory age or oldest retained evidence age
- lead/source stability summary
- path/curvature stability summary, initially shadow-only if lateral is not active
- map/speed/source stability summary where mode-eligible
- invalid or stale evidence counters
- legacy-equivalent/delta metrics while milestones are non-authoritative

Milestone 1 telemetry should make duplication visible: it should report which existing artifact supplied each summarized fact rather than hiding a new parallel classification behind the same labels.

## Validation gate

The **Planner Validation Gate** requires at least:

- unit tests for `PlannerStack` manifest/resolution/default behavior
- tests proving unavailable/unvalidated `scene-memory-v1` resolves to `planner-current`
- append-only cereal/schema contract tests for dedicated telemetry fields
- UI tests proving normal cruise settings cannot select unvalidated active `scene-memory-v1`
- fail-closed contract tests for invalid future active scene-memory output
- deterministic replay or Drive Lab comparison against `planner-current`
- at least one route-derived regression covering lead/source, stop, or speed/map behavior before active source changes are promoted
- equivalence tests proving validation-gated `scene-memory-v1` preserves `planner-current` longitudinal output fields where practical

## Milestone 1: durable boundary and shadow Scene Memory

Goal: complete the production-safe boundary and a shadow Scene Memory snapshot without changing control behavior.

### M1a: PlannerStack boundary

Deliverables:

1. Add `PlannerStack` param metadata with default `planner-current`.
2. Add planner-stack manifest and resolver.
3. Add `planner-current` and `scene-memory-v1` constants.
4. Add validation-gate handling so unvalidated `scene-memory-v1` resolves to `planner-current`.
5. Add dedicated planner-stack telemetry schema and publish helpers.
6. Add a normal cruise-settings Planner Stack row, gated so active `scene-memory-v1` cannot be selected before validation passes.
7. Add tests for resolver, validation gate, UI gating, telemetry defaults, and latching/default behavior.

### M1b: Shadow Scene Memory snapshot

Deliverables:

1. Add dedicated scene-memory telemetry schema with scaffold/default values.
2. Add a `SceneMemory` scaffold that records volatile freshness, confidence, source eligibility, and continuity from existing artifacts.
3. Consume existing artifacts instead of reclassifying them: `PrimaryLeadContext`, `SccEvidenceResult`, `LongitudinalCandidate`/decision telemetry, speed-limit/map/SCC provider state, and current planner output.
4. Publish source provenance for each summarized scene-memory fact.
5. Keep all longitudinal output legacy-equivalent while validation-gated; no actuation change.
6. Keep `planner-current` as the actuated planner stack even when a test/dev validation gate resolves `scene-memory-v1` for shadow telemetry.
7. Add tests for shadow memory defaults, artifact consumption, source-boundary labeling, telemetry publication, and planner-current equivalence.

Milestone 1 is complete only when tests pass and `planner-current` behavior remains unchanged.

## Relationship to CustomV2Scene

`CustomV2Scene` is already a stack-local normalized scene snapshot for custom-2.0. `SceneMemory` must not evolve overlapping lead/progress/source semantics independently.

Preferred future direction: `SceneMemory` becomes the planner-family snapshot and `CustomV2Scene` is derived from it where custom-stack policy still needs stack-local inputs. Until that migration is explicit, `SceneMemory` may only summarize existing `CustomV2Scene`-like artifacts for telemetry and validation.

## Later milestones

### Milestone 2: shadow all-source scene memory

- Track all mode-eligible longitudinal source families.
- Publish candidate/delta telemetry against `planner-current`.
- Keep output legacy-equivalent.
- Build Drive Lab comparisons and route regressions.
- Begin deriving custom-stack-local scene snapshots from Scene Memory only after equivalence tests prove no semantic drift.

### Milestone 3: active longitudinal source arbitration

- Promote specific source families from shadow to active after route/regression evidence.
- Preserve mode boundaries and MPC feasibility.
- Use fail-closed validation for invalid active output.

### Milestone 4: full longitudinal planner rewrite

- Move source interpretation, candidate arbitration, and longitudinal plan synthesis into `scene-memory-v1`.
- Keep existing controllers and MPC physical feasibility boundary unless a later ADR changes that boundary.

### Milestone 5: lateral shadow planning

- Add lateral planner-family shadow telemetry only after longitudinal milestones are stable.
- Lateral memory remains advisory until separate lateral validation exists.

## Branch ownership

This plan belongs on `feat/longitudinal-control` until lateral authority becomes real. Future lateral-planner authority work belongs on `feat/lateral-control`, with shared planner-stack vocabulary remaining in the longitudinal context unless a new planner-wide context is introduced.
