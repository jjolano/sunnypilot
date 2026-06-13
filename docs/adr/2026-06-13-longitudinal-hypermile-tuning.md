# Longitudinal hypermile tuning (Phase 5 feel)

Status: proposed
Date: 2026-06-13
Relates to: [clean-room longitudinal architecture](2026-06-13-clean-room-longitudinal-architecture.md),
legacy ADR `docs/legacy/adr/0001-longitudinal-planner-mpc-boundary.md`, the
`profile_lead_following` drive_lab metric (commit `68f3c5ab12`), and `docs/touch-points.md`.

## Context

The owner wants hypermile-style lead-following: follow with less hang-back, **anticipate** lead
motion rather than reacting, keep approaches low-jerk, and pull away promptly from a known/stopped
lead — without ever spending a speed-up that needs hard braking to recover. (See the hypermile
preference captured for this fork.)

Measured baseline (`profile_lead_following` over the device's engaged routes; the following corpus
is essentially `000001c7`, ~3 min of engaged cruise-following with a lead, plus a smaller pooled set):

- **Steady headway** `dRel/vEgo` ≈ **1.8 s** (route `c7`; ~1.9 s pooled), gap ≈ 48 m; **20–43%** of
  steady following sits above 2.0 s.
- **Approaches**: peak decel median −0.37, harshest-decile −1.6 m/s² (route) / ~−2.3 pooled; approaches
  *begin* ~2.5 s headway back (early, reactive).
- **~50% (route) / 37% (pooled) of slower-lead approaches ended with the lead speeding back up** — i.e.
  it decelerated for a transient lead slow-down that never became a stop.

The architecture constraint sorts the levers. Per the [architecture ADR](2026-06-13-clean-room-longitudinal-architecture.md)
and legacy ADR 0001, **custom longitudinal is policy arbitration over valid MPC envelopes — the MPC
keeps all physical lead-follow authority; the custom policy is a shaper that can only restrict, never
relax, the MPC's follow decel.** The planner runs the shaper *after* the MPC envelope is chosen
(`longitudinal_planner.py:82`, `custom_long.apply(... output_a_target ...)`). Therefore:

- the **desired follow gap is already `FOLLOW_TIME_GAP_S = 1.5 s`** (`stack.py:32`) — ≈ openpilot-normal
  (the `LEAD_CONTEXT_CLOSE_TIME_GAP = 2.2` the early analysis cited is a *risk-classification* threshold,
  not the follow gap). The measured ~1.8–1.9 s is mostly MPC/cushion behaviour, **not** a gap constant to
  lower (and lowering below ~1.4 s is explicitly out of bounds);
- the **reactive braking** (the most-felt issue) is MPC-owned and **cannot be un-braked by the shaper** —
  reducing it means shaping the lead-accel *fed to the MPC*, which touches the planner↔MPC boundary, not a
  custom constant.

## Decision

Adopt the hypermile change-set below, **sequenced easiest-first by tractability**, each gated on the
`profile_lead_following` metric over the engaged corpus (replay old vs new, diff the metric), with
property tests for the invariants in §Invariants. No safety cap moves.

### 1. Launch / pull-away from a known or stopped lead — *clean custom-policy win* (first)

The off-the-line hesitation is a trust+distance gate, all inside the custom policy
(`policy.py:173-179` `lead_pullaway`; `lead_context.py` `_close_stop_pullaway_progress_allowed` /
`_stopped_gap_creep_progress_allowed`). Tune, within the speed-up guard:

- shrink the stopped-gap creep arm toward 1.0× desired (`LEAD_CONTEXT_STOP_GAP_CREEP_ARM_EXCESS` 1.05 →
  ~1.0, keep `MAX_EXCESS` 1.25) so creep arms at the desired gap rather than well beyond it;
- shorten the confirmation hold for an **already-known** (not new / not flickering) lead so a confirmed
  re-launch isn't delayed (the new-lead `guard_timer` / `FALSE_POSITIVE_HOLD` stay for genuinely new leads);
- key the pull-away accel on the lead's actual opening (`v_lead` / `opening_speed`) rather than only on
  `gap_excess > 0`, so it follows the lead's launch instead of waiting for the gap to open first.

`lead_speedup_guard` (`lead_cushion.py`, cap −1.2 m/s²) is **unchanged** — it remains the floor that
prevents over-committing to a speed-up.

### 2. Approach coast / cushion shaping — *custom advisory, mixed* (second)

The early lift-off that adds hang-back and shapes the approach lives in the custom advisory cap
(`coast_horizon.py` `COAST_ARRIVAL_MARGIN_S = 0.6`, `lead_cushion.py` `lead_following_cushion`). Tune the
arrival margin / cushion comfort-brake so the approach lifts a touch later and bleeds speed more smoothly
(lower decel *peak*, same total), letting steady headway settle nearer 1.5–1.6 s. This only ever softens an
advisory cap toward — never through — the MPC envelope, and stays above the comfort floors.

### 3. Lead-motion anticipation / reactive braking — *MPC-boundary, hardest* (last; may spin its own ADR)

The metric machinery exists but is under-fed: `predict_lead_trajectory` (`lead_prediction.py:36`) takes an
`accel_confidence ∈ [0,1]` that discounts the noisy `aLeadK`, but it **defaults to 1.0 and the call site
doesn't compute it**, so transient lead-decel spikes propagate at full weight (decayed only by
`A_LEAD_TAU ≈ 1.5 s`). Two levers: compute `accel_confidence` from lead track stability/age and pass it;
and/or shorten `A_LEAD_TAU_DEFAULT` so transients decay faster. **But** this prediction currently only gates
*progress* — the binding follow decel that produces the reactive braking is the MPC's. Real impact needs the
confidence-/tau-shaped lead-accel to reach the **MPC input** (planner lead processing), not just the custom
progress gate. Because that touches the planner↔MPC boundary (ADR 0001), it is sequenced last and, if it
proves to require boundary changes, gets its own ADR rather than riding this one.

## Invariants (must not move)

- MPC physical lead-follow authority binds (ADR 0001 / architecture ADR §1); the shaper only restricts.
- `STOP_APPROACH_DECEL_MIN = −1.5 m/s²` (`policy_tables.py:55`) and `SPEEDUP_GUARD_MAX_REQUIRED_DECEL =
  −1.2 m/s²` (`lead_cushion.py:27`) — comfort/safety floors, unchanged.
- Accel limits `[−4.0, +2.0]` (`wiring.py:27`); model-stop `TRUST_FULL_STOP = 0.7` (`model_trust.py`).
- Fail-closed; mode admissibility (`modes.py`) respected. Personality tables scale comfort/progress only,
  never safety caps (architecture ADR §4).

## Consequences

- Order of work matches risk: §1 lands first (cleanly custom, property- + replay-testable), §2 second,
  §3 last (and possibly deferred to a boundary ADR).
- The `profile_lead_following` metric is the feel gate: success = headway settling toward ~1.5–1.6 s,
  reactive-brake share and decel-peak down, launch hesitancy down — with the invariants intact. As the
  architecture ADR notes, this hypermile feel is the most engaged-data-gated work; default-on waits on
  replay, like the torque governor.
- Following data on the current corpus is thin (~3 min); widening it (more lead-present engaged routes) is a
  prerequisite for trusting the §2/§3 deltas, and `bd`/`be` should be re-pulled cleanly.
- Custom-policy changes go in new files / hook-sized diffs per the repo model; any planner-lead-input change
  for §3 is a touch-point to record in `docs/touch-points.md`.
