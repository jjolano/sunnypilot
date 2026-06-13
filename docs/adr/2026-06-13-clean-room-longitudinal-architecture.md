# Clean-room longitudinal architecture (Phase 4)

Status: proposed
Date: 2026-06-13
Relates to: [restart plan Phase 4](../plans/2026-06-12-fork-restart-reimplementation.md),
legacy ADRs `docs/legacy/adr/0001-longitudinal-planner-mpc-boundary.md`,
`docs/legacy/adr/2026-05-17-custom-longitudinal-v2-architecture.md`,
`docs/legacy/adr/2026-05-31-longitudinal-modes.md`, and
`docs/legacy/CONTEXT-longitudinal.md`.

## Context

custom-2.0 longitudinal is the anticipatory / hypermile feel the owner wants to keep. The
legacy implementation is ~3.4k lines split across a decision core (`longitudinal_decision.py`
891) and a policy stack (`custom_v2.py` 1051), plus deterministic models (`lead_context.py`
853, `lead_confidence.py` 179, `custom_v2_trajectory.py` 72) and seed mapping
(`planner_seed_policy.py` 330, `planner_seed.py` 209). It carries two debts: a
decision-core/policy **duality** (`_decide` vs `_decide_from_core`) and **scene-struct
shadowing**. Per the 2026-06-13 review, the ACC/E2E/SCC **mode** layer is a kept product
feature.

## Decision

### 1. Preserve the boundary (ADR 0001, verbatim)

Custom longitudinal is **policy arbitration over valid MPC envelopes**. The MPC keeps all
physical lead-follow authority (closing-rate, danger-gap, stop-runway feasibility). Custom
policy chooses among valid envelopes; it is not a competing lead-physics model. Faults are
**fail-closed** (request disable, never silently fall back after authority begins).

### 2. Module layout

```
sunnypilot/custom/longitudinal/
  modes.py            # ACC/E2E/SCC evidence-admission gate (NEW, this increment)
  lead_confidence.py  # faithful port (flicker/continuity)
  lead_context.py     # faithful port (risk/progress model)
  trajectory.py       # faithful port (jerk-limited synthesis)
  seeds.py            # planner seed -> intent mapping (port)
  decision.py         # unified decision core + policy (the merge)
  policy_tables.py    # personality-as-data
```

### 3. Longitudinal modes — the outer evidence-admission gate (built first)

A **Longitudinal Mode** is the top-level user choice; it decides which *classes of evidence*
may reach actuation **before** policy candidates are built. It is latched per onroad cycle,
sits above the policy overlay, and nothing downstream re-admits what the mode excluded
(CONTEXT-longitudinal: SCC Mode / SCC Curve Control). Owner intent (2026-06-13):

- **ACC** — OEM-like cruise: admits `CRUISE` + `LEAD` only. Excludes model-stop, map,
  speed-limit, OSM, and curve evidence.
- **E2E** — the model drives: admits the full model set incl. `MODEL_STOP` (traffic
  lights / stop signs the model detects).
- **SCC** — intelligent ACC/E2E blend: ACC-like base + the model's traffic-control
  awareness (`MODEL_STOP`), plus curve sources gated by `SccCurveVisionEnabled` /
  `SccCurveMapEnabled`, plus map/speed-limit within mode boundaries.

`modes.py` is one pure function `admitted_evidence(mode, sources) -> frozenset[EvidenceClass]`
plus the enums. It is the ONLY place admissibility is decided. Property-tested exhaustively
(ACC excludes traffic-control/map/curve; E2E admits model-stop; SCC curve sources follow
their toggles) — no engaged data needed for this gate's correctness.

### 4. Decision core + policy — the merge (behavior-changing, engaged-gated)

Merge `longitudinal_decision.py` + `custom_v2.py` into one `decision.py`: the candidate /
intent / authority model is kept; the `_decide` legacy path is deleted (`_decide_from_core`
becomes the only path); scene-struct shadowing is removed in favor of reading lead/SCC/
planner evidence directly. Personality becomes declarative tables (`policy_tables.py`):
launch accel, stop-approach decel, coast-leeway bounds, comfort-relax floor, jerk budgets —
standard personality anchors, others scale comfort/progress only, never safety caps. This
merge changes behavior and is **gated on the engaged-route corpus** (replay old custom-2.0
vs new, diff per-scenario decel profiles / launch timing / coast leeway). Property tests
gate invariants (safety caps only restrict; fail-closed; mode admissibility respected); they
do not certify feel.

### 5. Faithful ports (deterministic, parity-portable now)

`lead_confidence.py`, `lead_context.py`, `trajectory.py`, and the seed mapping are clean
deterministic modules; they are relocated faithfully (like the Phase 3 demand processors)
with behavioral tests, not rewritten blind. They carry the risk/progress/jerk knowledge the
policy consumes.

## Consequences

- The mode gate is buildable and fully testable now (this increment); it realizes the
  kept ACC/E2E/SCC product feature with a single admissibility authority.
- The decision-core merge — the hypermile feel — is the most engaged-data-gated work; its
  structure can be built and invariant-tested, but feel-cert and default-on wait for engaged
  replay, exactly like the torque governor.
- SCC curve sources depend on the Phase 6 map/OSM work; until it lands, SCC blends
  model-stop/lead evidence and the curve toggles are inert (admitted set simply omits them).
- Map/OSM and speed-limit evidence providers are Phase 6; the mode gate already names their
  evidence classes so wiring them later is additive.
