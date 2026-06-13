# Fork Restart: Reimplementation Plan

Status: proposed
Date: 2026-06-12

Restart the fork from upstream sunnypilot, reimplementing only the proven-valuable behavior
(torque v2.1 lateral, custom-2.0 longitudinal policy) as cleaner rewrites, and dropping the
multi-stack scaffolding. The rewrite doubles as the experiment that attributes the known
v2.1 quirks to controller vs planner, and as the opportunity to land the four unimplemented
improvement specs.

## Goals

1. Keep the torque v2.1 driving feel (responsiveness) and the custom-2.0 anticipatory /
   hypermile longitudinal feel.
2. Resolve whether the v2.1 quirks are controller-side or demand/planner-side.
3. Shrink the upstream diff from ~95k lines to a target of ~15k, structured for cheap rebases.
4. Improve, not just port: unify the lateral output stage, collapse the longitudinal decision
   duality, and land the backlog improvement specs.

## Non-goals

- Preserving the stack-selector / registry / promotion-gate machinery (single known-good
  implementation per axis, plain params for toggles).
- Porting torque v0/v3/v4, custom-experimental stacks, or scene-memory-v1.
- One-pedal longitudinal mode — dropped (not used day-to-day; confirmed 2026-06-13).
- Tailscale IS kept — ported 2026-06-13 as new files plus three small touch points; see
  `docs/touch-points.md`.

SCC mode is KEPT, not deferred — see "Longitudinal modes" below.

## Repo model (simplified)

The old fork's branch machinery (feature branches octopus-merged into `custom`,
retained-baseline snapshots, `.sync-config`, propagate/sync scripts) is retired. The new
model:

- **One history**: work happens as linear commits on a single branch built directly on
  `master` (which tracks upstream). The `restart` branch fast-forwards onto `master` when
  ready; after that, `master` is the fork.
- **Upstream updates**: merge `upstream/master` into the branch when deliberately chosen —
  no scheduled sync, no propagation scripts. The small-touch-point discipline below is what
  keeps those merges cheap.
- **Reference**: the old `custom` branch is frozen and kept as the executable reference
  implementation; nothing is developed there. Quirk forensics on old route logs run from a
  `custom` checkout, since those logs carry the old cereal schema's custom fields.
- **No stack multiplexing**: features are plain params, one implementation per axis.
- **Deploy workflow retained**: the old fork's deploy script survives in simplified form
  (`scripts/deploy.sh` + `.deploy-config` + the `deploy-workflow` agent skill). The
  "commit rebuild deploy" verb interface keeps working; "rebuild" (re-assembling the
  integration branch) is now a no-op since there is no integration branch to assemble.
- **Agent guidance**: `AGENTS.md` is the canonical agent-instructions file; `CLAUDE.md`
  is a symlink to it.

## Guiding principles

- **Behavior before structure.** Every port phase ends with a drive_lab comparison against
  baselines captured from the *current* fork (Phase 0). Feel regressions are detected with
  data, not memory.
- **One layer at a time.** Lateral controller lands and is driven before any demand-side
  code is ported. This is what isolates the quirk.
- **Constants are the asset.** All tuned constants are extracted into a single reference
  inventory before the restart; rewritten modules consume them from declarative tables, not
  scattered literals.
- **Minimal touch points.** Custom code lives in new files; diffs to upstream files are
  hook-sized and individually documented (the existing `_ext` pattern, applied with
  discipline). Pin the upstream base to a tag; rebase deliberately, not continuously.

---

## Phase 0 — Capture knowledge and evidence (on the CURRENT fork, before anything else)

The current fork is the only executable specification of the feel worth keeping. Capture it
while it still drives.

1. **Port drive_lab to the new branch wholesale** (`tools/drive_lab/`, ~15k lines). Its
   only external imports are upstream modules (`common.constants`, `common.prefix`,
   `tools.lib.logreader`), so it carries over as-is and becomes the measuring instrument
   for the whole migration. Keep `route_io.py`, `metrics.py`, `timeline.py`, comparison
   tools, and the gates (`behavior_change_gate.py`, `lateral_performance_gate.py`).
   Caveat: analyses that read the old fork's custom log structs must run from the frozen
   `custom` checkout, which has the matching cereal schema.
2. **Build the route corpus.** Select 8–12 representative logged routes covering: low-speed
   city (under-response floor regime, <12 m/s), highway cruise, curvy roads, lead-follow
   stop&go, slowing-lead approaches, downhill coast sections, and — critically — every route
   where a known v2.1 quirk occurred.
3. **Bake baselines.** For each corpus route, run and archive:
   - `profile_lateral_performance.py` / `lateral_event_report.py` / `lateral_oscillation_profile.py`
   - `profile_route.py` / `planner_target_analysis.py` longitudinal profiles
   - `manual_longitudinal_baseline.py` outputs for routes with manual-driving segments
     (the hypermile reference is *your* driving, not the old code).
4. **Pre-restart quirk forensics.** On the quirk routes, attribute each event using the
   diagnostics that already exist:
   - Demand side: `lateralDemand` debug — `demand_source`, `model_path_reason`,
     `raw_curvature` vs `processed_curvature`, lane-centering nudge magnitude.
   - Controller side: `adaptiveTorqueState` — `governorReason`, `shapingReason`,
     `disturbanceState`, `outputCap`, assist phase.
   If the quirk shows up in `processed_curvature` before the controller runs, it is
   demand/planner-side; if processed curvature is clean and output torque misbehaves, it is
   the controller's output stage. Record verdicts per event — this directly shapes Phase 2/3
   scope.
5. **Extract the constants inventory.** Script a sweep of the kept modules
   (`latcontrol_torque_v2.py`, the torque helpers, `custom_v2.py`, `planner_seed_policy.py`,
   `lead_context.py`, `custom_v2_trajectory.py`, `longitudinal_decision.py`) into one
   reviewed YAML: name, value, unit, source file:line, behavioral meaning. ~150 constants.
6. **Copy the paper assets** into the new home: `CONTEXT.md`, `docs/adr/`, `docs/concepts/`,
   the four backlog specs, and the quirk-forensics notes from step 4.

Exit criteria: corpus + baselines archived outside the repo; constants YAML reviewed;
quirk events have per-event controller/planner verdicts (or are explicitly "unresolved,
test in Phase 2").

## Phase 1 — New base

1. Branch `restart` from `master` (== upstream/master at `01a843e0ac`). Linear commits;
   fast-forward onto `master` when ready.
2. Land the docs from Phase 0.6 (under `docs/legacy/` for imported reference material) and
   the constants YAML.
3. Define the layout: all custom behavior under `sunnypilot/custom/` (controller, longitudinal
   policy, shared math), new files only; a `docs/touch-points.md` listing every modified
   upstream file with one line of why. Target: <10 modified upstream files at end state.
4. Cereal: start with zero custom fields. Add fields only when a kept feature demands them,
   into one `custom.capnp` section per subsystem. (Old fork: +262 lines, mostly multi-stack
   diagnostics — expect <80 lines.)
5. Tooling: simplified `scripts/deploy.sh` (+ `.deploy-config`, `deploy-workflow` skill)
   ported from the old fork, minus the rebuild/propagate machinery. No sync scripts.
6. `AGENTS.md` (with `CLAUDE.md` symlink) documents the repo model, workflow keywords,
   testing setup, and deploy health checks for agents.

## Phase 2 — Lateral controller: torque v2.1 as a unified rewrite

**Integration: join upstream's native selector, don't rebuild the fork's.** Upstream
sunnypilot already ships a torque controller list (`TorqueControlTune` param,
`latcontrol_torque_versions.json` manifest driving the torque settings UI and sunnylink
schema, dispatch in `controlsd_ext.initialize_lateral_control()`) with v0.0 and v1.0
entries. The rewritten v2.1 becomes a third entry: one manifest line, one new controller
module, one dispatch case in `controlsd_ext.py` (a designed extension point — small
touch). The list reads v0.0 / v1.0 / v2.1; both stock fallbacks come free. Default is
`TorqueControlTune=2.1` on the device once this phase validates. The fork's parallel
`torque_versions.py` registry and the retired versions (v2.0, v3, v4.x, 5.0) do not
return. Longitudinal has no upstream equivalent (`LongitudinalStack` was fork-invented),
so it stays a single fork param — custom policy on/off — defaulting on after Phase 4.

**Clean-room architecture, preserved behavior.** The controller is rebuilt with whatever
architecture is best — it is not a file-by-file port. What is preserved is *behavioral*:
the v2 response math that carries the feel (speed-interpolated KP schedule, 1 s
lateral-accel request buffer + delay compensation, measurement smoother, low-speed
unwind, low-demand friction scaling) and the tuned constants, with parity enforced by
the Phase 0 corpus gates rather than by structural fidelity. Restructure freely; change
behavior knowingly (each intentional behavior delta gets named and gated) or not at all.

**Rewrite** the output stage as part of that. Today four mechanisms stack with
cross-layer special cases (over-response attenuator → guarded response assist →
conservative shaper → refined governor, with e.g. `shaper_already_capped` leaking shaper
state into governor gating). This stack is the prime controller-side quirk suspect.
Replace with **one** output governor:

- Single input observation struct (the existing `TorqueObservation` generalized).
- Single pass computing: cap (high-rate, same-direction, saturation context), slew
  (speed-scheduled, sign-change, high-rate scaling), floor (under-response, low-speed
  catch-up), assist/release phases — in one explicit precedence order.
- One reason bitfield, one log struct. Every intervention attributable in a single field.
- Property tests: output continuity under input steps, cap monotonicity, floor/cap
  interaction (floor may only restore toward unclipped, never exceed `max_output`),
  sign-change behavior. Port the relevant cases from `test_latcontrol_torque_v2.py`,
  `test_torque_conservative_output_shaper.py`, `test_torque_guarded_response_assist.py` as
  behavioral fixtures against the unified implementation.

**Validate:**
1. Offline: replay corpus routes through old v2.1 and new controller
   (`compare_lateral_torque.py` / `compare_torque_versions.py` adapted); gate on baseline
   deltas for tracking error, oscillation score, intervention rates.
2. On-road against **stock planner output** — no demand stack. This is the quirk experiment:
   - Quirks gone → they were demand-side or layer-interaction; Phase 3 proceeds cautiously.
   - Quirks remain → controller-side; fix inside the unified governor where the reason
     bitfield now says exactly which mechanism fired.

Optional sub-feature (port as-is, param-gated, default off until validated): speed-adaptive
torque learning (`speed_aware_torque.py`, `torqued_ext.py`, ~250 lines + small `torqued.py`
hook).

Exit criteria: corpus gates pass; ≥1 week daily driving with feel parity; quirk verdict
recorded per Phase 0 event.

## Phase 3 — Lateral demand layer: evidence-driven, piecewise

Only port pieces that Phase 0/2 evidence says carry value, one at a time, each behind its
own param, each validated on the corpus before the next:

1. `model_path_processor.py` (smoothing/quality gating) — most likely valuable.
2. `lane_change_path_shaper.py` — standalone, port if used.
3. Lane-centering assist nudge — port last; curvature nudges are a classic quirk source.

Improvement: no demand "stack" abstraction. A single `LateralDemandPipeline` with ordered,
individually-toggleable processors and one debug struct (`demand_source`, per-processor
delta on curvature). The pipeline logs raw vs post-each-processor curvature so future quirk
attribution is a log query, not a fork restart.

Exit criteria: each enabled processor shows a measurable corpus win (or is dropped);
combined feel parity with old fork on the corpus.

## Phase 4 — Longitudinal core: modes + policy overlay, decision duality collapsed

**Preserve the architecture** (ADR 0001 verbatim): custom longitudinal is policy arbitration
over valid MPC envelopes. MPC keeps all physical lead-follow authority. Fail-closed faults.

### Longitudinal modes (ACC / E2E / SCC) — KEPT

This is a deliberate product feature, not stack scaffolding, and it is preserved. A
**Longitudinal Mode** is the top-level user behavior choice; it gates which *classes of
evidence* may reach actuation before policy candidates are built. The three modes and their
intended character (owner intent, 2026-06-13):

- **ACC** — a faithful OEM-like adaptive cruise: speed hold + lead following only. It must
  *not* consume model-stop, map, speed-limit, OSM, or curve evidence. The bar is "feels like
  the car's factory ACC", so a driver can pick it and get predictable, boring cruise.
- **E2E** — the driving model drives the car: full model authority, including reacting to
  traffic lights and stop signs the model detects. This is the headline advantage of E2E
  over ACC.
- **SCC** — intelligently blends ACC and E2E: ACC-like smoothness and predictability as the
  base, with E2E's traffic-control awareness (lights, stop signs, stops) layered in inside
  mode-specific boundaries. SCC selects ACC-like or E2E-like behavior from explicit evidence
  rather than running raw model output.

**Joint control toggle** is kept: one convenient setting switches the active mode (the old
fork's mode selector UX). Modes are a latched per-onroad-cycle choice, not a per-tick
arbiter, and they sit *above* the policy overlay — the overlay still arbitrates within
whatever evidence the active mode admits.

**Rewrite shape (mode layer):** a single `LongitudinalMode` enum + an evidence-admission
gate (one function: mode → allowed evidence classes), replacing the old DEC/mode plumbing.
The gate is the *only* place that decides admissibility; the policy overlay and MPC never
re-admit evidence the mode excluded. SCC's blend logic is a named, tested component, not
scattered conditionals. Reference: legacy ADR `2026-05-31-longitudinal-modes.md` and the
SCC Mode / SCC Curve Control language in `docs/legacy/CONTEXT-longitudinal.md`. SCC curve
sources (`SccCurveVisionEnabled`, `SccCurveMapEnabled`) come with the map work in Phase 6;
until then SCC blends model-stop/lead evidence only and the curve-source toggles are inert.

**Rewrite shape (policy overlay)** (~2.5k lines target, from ~3.4k):
- Merge `longitudinal_decision.py` (891) and `custom_v2.py` (1051) into one decision core +
  one policy module. Delete the legacy `_decide` path — `_decide_from_core` becomes the only
  path. The candidate/intent/authority model is good; the duality and scene-struct shadowing
  are the debt.
- `lead_context.py` (853) ports mostly as-is — it is the risk model and it works. Trim the
  scene-memory shadow fields.
- `lead_confidence.py` (179) and `custom_v2_trajectory.py` (72, jerk limits) port as-is.
- `planner_seed_policy.py` (330): keep the seed/intent mapping; seeds remain planner-owned.
- **Personality as data**: replace the ~72 scattered constants with one declarative policy
  table per personality (launch accel, stop-approach decel, coast leeway bounds, comfort
  relax floor, jerk budgets), loaded from the Phase 0 constants YAML. Standard personality
  anchors; others scale comfort/progress only, never safety caps (existing rule, now
  enforced by table schema).

**Validate:** corpus replay with `planner_target_analysis.py` + `compare_manual_planner_targets.py`
against Phase 0 baselines; `fuzz_longitudinal.py` and the scenario simulator ported as the
regression harness; then on-road parity weeks like Phase 2.

Exit criteria: per-scenario corpus gates (slowing-lead approach decel profile, stop-approach
comfort, launch/pullaway timing, downhill coast leeway) within tolerance of old-fork
baselines; no safety-cap behavior changes.

## Phase 5 — Longitudinal improvements (the "room for improvement")

Now, on the clean core, land the backlog in order of hypermile impact:

1. **Coast-horizon anticipation** (new design work — the principled upgrade to the current
   heuristics). Compute a physics-based coast-down trajectory from current speed, road grade
   (existing pitch input), and a learned rolling drag estimate; choose lift-off points so
   speed bleeds to the constraint target (curve cap, speed limit change, slowing lead's
   projected speed) exactly at the constraint, instead of leeway-band heuristics. This
   subsumes parts of free-coast and comfort-relax. Needs a short ADR + spec first; validate
   against the manual-driving baselines — the metric is "matches the human lift-off point".
2. **Lead-following cushion** (spec exists, 2026-05-03): coast-first taper on normal-speed
   closing leads — integrates naturally with (1).
3. **E2E runway comfort governor** (spec exists, 2026-05-03).
4. **Map-curve soft advance** (spec exists, 2026-05-05) — only if map curve sources are
   re-enabled (decision point below).
5. **Lead-aware speedup guard** (spec exists, 2026-05-05): cap pullaway progress by actual
   gap + required decel rather than gap-excess prediction.

Each lands as its own change with its own corpus gate. Drag estimate learning in (1) is the
only new learned state; persist it like the existing live torque params.

## Phase 6 — Optional periphery (only what real usage justifies)

Decision points — defer until the core phases prove out:
- Map/OSM curve + speed-limit sources (old speed-map-control work): wanted, because they
  feed SCC mode's curve awareness (`SccCurveVisionEnabled` / `SccCurveMapEnabled`) and the
  Phase 5 map-curve soft advance. The mapd plumbing is large, so it lands here rather than
  blocking the core; SCC mode ships functional without it (traffic-control/stop awareness
  from the model) and gains curve sources when this lands.
- Speed-adaptive torque learning promotion to default-on (from Phase 2 optional).
- UI settings pages beyond the minimal params needed.

Dropped (not deferred): one-pedal longitudinal mode. Tailscale already landed (Phase 1).

## Validation infrastructure (cross-cutting)

- drive_lab (in-tree) is the gate for every phase; wire `behavior_change_gate.py` +
  the performance gates into CI so a change cannot land without corpus checks.
- Keep the old fork installable on the device throughout; A/B by swapping branches, never
  by losing the reference implementation.
- Port tests only for kept modules, rewritten against the new shapes; target is behavioral
  fixtures from real-route data over synthetic unit minutiae.

## Risks

- **Feel regression invisible to metrics.** Mitigation: parity driving weeks per phase plus
  manual-baseline comparisons; keep old fork bootable for instant A/B.
- **Unified governor changes quirk behavior before attribution.** Mitigation: Phase 0
  forensics happen on the old fork first; Phase 2 on-road step is explicitly the experiment.
- **Upstream drift during the migration.** Mitigation: pinned base tag; one deliberate
  rebase after Phase 4, using the touch-points doc.
- **Scope creep back toward multi-stack.** Mitigation: non-goals above; params not stacks;
  any second implementation of an axis requires a new ADR.

## Sequencing summary

| Phase | Output | Gate |
|---|---|---|
| 0 | drive_lab port, corpus + baselines, constants YAML, quirk verdicts, docs | baselines archived |
| 1 | `restart` branch on master, layout, docs landed | — |
| 2 | unified torque v2.1 controller | corpus parity + quirk experiment |
| 3 | demand pipeline (piecewise) | per-processor corpus win |
| 4 | merged longitudinal decision core + policy tables | per-scenario parity |
| 5 | coast-horizon + backlog specs | beats baseline on efficiency/comfort metrics |
| 6 | optional periphery | per-feature |
