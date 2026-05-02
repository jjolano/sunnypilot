# Longitudinal Decision Layer Design

## Summary

Longitudinal control has accumulated several branch-owned features that all shape speed or acceleration: lead following, lead transition handling, e2e stop approach, cruise coasting, launch, speed-limit assist, SCC vision/map control, OSM traffic-control priors, and FCW gating. The current architecture makes these features hard to reason about together because independent policies collapse into loose scalar targets early, then planner and MPC code apply more stateful overrides afterward.

This design introduces a feature-flagged longitudinal decision layer that normalizes feature outputs into explicit candidates, applies one central arbitration policy, publishes rich telemetry, and falls back to the existing planner output whenever the new path is disabled or invalid.

## Goals

- Unify longitudinal feature interactions through one explicit decision model.
- Support behavior redesign, not just a mechanical refactor.
- Preserve a safe fallback to the current planner while the new layer is validated.
- Make every slowdown or acceleration suppression explainable through telemetry.
- Keep actuator-side `LongControl` behavior isolated from planner-level arbitration.
- Add a UI settings toggle, default off, so the new behavior is opt-in at first.

## Non-Goals

- Do not rewrite `LongControl` as part of the first decision-layer implementation.
- Do not remove the existing MPC solver or replace Acados.
- Do not make map-only traffic controls command full stops.
- Do not treat `custom` as the source of truth for this work.
- Do not collapse all retained longitudinal branches into one branch without updating workflow ownership.

## Current Architecture Problems

The current pipeline roughly works as follows:

1. Sunnypilot extension logic selects a lowered target from cruise, speed-limit assist, SCC vision, SCC map, and OSM/map features.
2. Core planner feeds that target into MPC and applies planner-level behavior such as e2e blending, engage bootstrap, cruise coast, and stopped-gap handling.
3. MPC builds lead/cruise obstacles and applies lead-following, lane-exit, crawl, and comfort logic.
4. `LongControl` consumes the final planner acceleration and stop intent.

The main issue is not that there are many features. The issue is that the feature boundary is implicit. Some policies lower `v_cruise`, some mutate planner output, some live inside MPC, and some are controller-side. This makes overlapping cases hard to predict, such as a close lead during a speed-limit reduction, a map curve while overspeed, or a low-confidence cut-in while the driver set speed remains high.

## Desired Driving Behavior

The redesigned behavior should optimize for efficient progress by default while preserving comfort and confidence gates where they matter.

- Default personality: efficient progress.
- Confirmed lead following: maintain a strict comfort gap.
- Cut-ins and newly detected close leads: wait for confidence before strong reaction, while preserving hard safety fallbacks.
- Lead departure or brief disappearance: release smoothly without surging.
- Stops behind leads or model-predicted stops: feel like a normal attentive human stop.
- Launch from stop: match the lead's motion, avoiding both lag and surge.
- No-lead overspeed cruise: be context efficient; coast when harmless and brake when speed excess affects curves, limits, or traffic.
- Speed-limit assist: act as an advisory cap, subordinate to active traffic and confirmed lead behavior.
- SCC vision/map curve slowdowns: require confident curves to avoid nuisance slowdowns.
- OSM traffic controls: map data may trigger mild caution, but stronger stopping requires model or lead confirmation.
- Disagreement between features: prefer driver intent unless a physical lead or high-confidence stop threat exists.
- Debuggability: expose rich telemetry for candidates, winners, suppressed candidates, confidence, urgency, limits, and fallback reasons.

## Proposed Architecture

Introduce a planner-level decision layer between raw feature logic and final `longitudinalPlan` publication.

The main units are:

- `LongitudinalCandidate`: a normalized proposal from one feature or policy source.
- `LongitudinalDecision`: the final plan selected by arbitration, plus telemetry.
- Candidate producers: adapters that convert existing cruise, MPC, e2e, speed-limit, SCC, OSM, coast, and stop/launch signals into candidates.
- `LongitudinalArbiter`: central policy that chooses or shapes the final decision from candidates.
- Feature flag param: persistent opt-in control for whether the decision layer drives output.
- Settings toggle: UI control for the feature flag, default off.
- Telemetry publisher: debug data that explains what happened and why.

The first implementation should be planner-level. `LongControl` should continue to receive final `aTarget` and `shouldStop` through the existing publish path. That keeps actuator behavior separate from planner arbitration.

## Candidate Model

Each candidate should describe both its target and why it should matter.

Required fields:

- `source`: stable enum or string such as `cruise`, `lead_mpc`, `e2e_stop`, `speed_limit`, `scc_vision`, `scc_map`, `osm_traffic_control`, `cruise_coast`, or `stop_launch`.
- `v_target`: candidate speed target in m/s.
- `a_target`: candidate acceleration target in m/s^2.
- `confidence`: normalized confidence that the source is correct.
- `urgency`: normalized urgency of acting now.
- `role`: candidate type such as driver intent, physical hazard, advisory cap, comfort shaping, or fallback.
- `comfort_bounds`: optional acceleration or jerk bounds for passenger comfort.
- `safety_bounds`: optional hard bounds required for safety.
- `should_stop`: whether this candidate wants a stop state.
- `active_reason`: concise machine-readable reason.
- `debug`: compact source-specific metadata for tests and logs.

The candidate model should be small and boring. It should make the policy explicit without becoming a second planner hidden inside data classes.

## Candidate Producers

Candidate producers wrap existing logic initially. They should avoid large behavior rewrites in the first pass unless the behavior change is necessary to express the policy correctly.

Initial producers:

- Cruise candidate: driver or PCM cruise setpoint and current cruise acceleration intent.
- Lead MPC candidate: MPC result for confirmed lead following, including source, crash counter, and lead confidence.
- E2E stop candidate: model-predicted no-lead slowdown or stop intent.
- Speed-limit candidate: advisory cap from speed-limit resolver and assist.
- SCC vision candidate: confident model curve slowdown target.
- SCC map candidate: map/advisory curve target with confidence gating.
- OSM traffic-control candidate: caution target from traffic-control priors.
- Cruise coast candidate: preference to avoid braking while harmlessly overspeed.
- Stop/launch candidate: stop and launch intent needed to preserve smooth stop-to-go transitions.

Candidate producers can be introduced incrementally. Early versions may adapt existing scalar outputs rather than moving every feature's internal logic immediately.

## Arbitration Policy

The arbiter should not simply choose the lowest target. It should classify candidates by role and apply the desired behavior policy.

Policy order:

1. Start with driver intent or cruise as the default candidate.
2. Apply confirmed physical hazards over driver intent when a lead or stop threat is sufficiently confident.
3. Allow high-urgency safety candidates to impose hard acceleration bounds.
4. Let advisory caps, speed limits, map curves, and OSM cautions shape the target only when confidence and context justify it.
5. Suppress low-confidence cut-ins, weak curve predictions, and stale map data unless another source confirms the threat.
6. Allow cruise coast to relax braking only when no safety or high-confidence advisory candidate requires deceleration.
7. Shape lead departures with a smooth release guard so acceleration does not surge after a lead exits or disappears.
8. For launch, prefer lead-matched acceleration over generic assertive launch when a lead is present.

When candidates conflict, the arbiter should emit both the winner and the suppressed candidates with reasons. This is required for debugging and future tuning.

## Data Flow

Runtime data flow:

1. Existing planner and sunnypilot feature logic computes raw signals.
2. Candidate producers convert raw signals into `LongitudinalCandidate` objects.
3. `LongitudinalArbiter` validates candidates, applies arbitration policy, and emits `LongitudinalDecision`.
4. If the UI toggle is off, the current planner output remains authoritative.
5. If the UI toggle is on and the decision is valid, the decision supplies final planner output fields such as target speed, target accel, source, and stop intent.
6. If the decision is invalid or unavailable, the planner falls back to current output and records a fallback reason.
7. Telemetry records candidates, decision, suppressions, and fallback state.

The old and new paths should stay comparable during rollout. Tests should be able to construct the same raw scenario and inspect both the legacy result and the decision-layer result.

## UI Toggle And Param

Add a persistent param for the new behavior, default off. The exact name can be finalized during implementation; `LongitudinalDecisionLayer` is the working name.

Settings behavior:

- Show a user-facing settings toggle for the new decision layer.
- Default the toggle to off.
- Label the behavior as experimental until validated on-device.
- When off, preserve current planner authority.
- When on, allow the decision layer to control final planner output subject to fallback validation.

The toggle and param are part of the retained feature scope for the decision-layer branch. If the branch becomes part of the retained workflow, update `.sync-config`, `AGENTS.md`, and any metadata docs together.

## Fallbacks And Error Handling

The existing planner output is the fallback authority.

Fallback rules:

- If candidate generation fails, use current planner output.
- If the arbiter raises an exception, use current planner output.
- If arbiter output is non-finite, stale, or outside hard acceleration bounds, use current planner output.
- If no candidate is valid except cruise, prefer driver intent unless a physical lead or high-confidence stop threat exists.
- If the UI toggle is off, use current planner output.
- Always record the fallback reason in telemetry.

The decision layer should fail closed to known behavior, not to an invented safe stop, unless existing planner logic already requests that stop.

## Telemetry

Rich telemetry is a core requirement, not an optional debug nicety.

Telemetry should include:

- Candidate list.
- Winning candidate or blended decision.
- Suppressed candidates.
- Suppression reasons.
- Confidence and urgency per candidate.
- Applied comfort and safety bounds.
- Fallback reason, if any.
- Whether the UI toggle was enabled.
- Legacy planner output for comparison where practical.

The first implementation may keep telemetry in Python-side structures and tests before adding cereal fields. If runtime telemetry needs to cross process boundaries or reach UI, schema changes should be designed carefully because `cereal/custom.capnp` is high-risk and branch-sensitive.

## Testing Strategy

Validation should happen in layers.

Required tests:

- Unit tests for candidate construction and validation.
- Unit tests for arbiter policy decisions.
- Tests that the feature flag off path preserves current planner output.
- Fallback tests for invalid, non-finite, stale, or exception-producing candidates.
- Overlapping-candidate tests covering lead plus speed limit, lead plus map curve, overspeed plus curve, OSM caution plus e2e stop, and low-confidence cut-in plus driver intent.
- Maneuver tests for cut-ins, lead departure, stopping, launch, no-lead e2e stops, cruise overspeed, and overlapping speed-limit/SCC/lead cases.
- Telemetry tests proving every decision includes a winner, suppressed candidates, confidence, urgency, source, and fallback reason when relevant.

Drive Lab should be used when real routes, bookmarks, or timestamps are available. Route-guided fuzzing should classify failures before behavior changes are made.

## Branch And Rollout Strategy

This work crosses existing retained branch scopes. The cleanest path is a new retained branch, tentatively `feat/longitudinal-decision-layer`, based after the existing retained longitudinal and adjacent branches.

Rollout phases:

1. Add design and branch ownership documentation.
2. Add candidate and arbiter types with unit tests.
3. Add adapters around existing feature outputs without changing runtime behavior.
4. Add feature flag and settings toggle, default off.
5. Run the arbiter in enabled mode only when the toggle is on; keep legacy fallback.
6. Add maneuver and overlapping-candidate coverage.
7. Validate on-device before considering default-on behavior.

If the branch is retained long-term, update `.sync-config` merge order and `AGENTS.md` ownership rules. Do not implement durable behavior only on `custom`.

## Open Implementation Decisions

- Exact param name and settings location.
- Whether candidate telemetry should initially be Python-only, logged, or added to cereal.
- Whether the arbiter should ever blend candidates or always choose a single winner plus bounds.
- How much existing MPC policy should remain inside `long_mpc.py` during the first implementation.
- Which maneuver scenarios are mandatory before on-device testing.

These are implementation planning questions. They do not block the design direction.
