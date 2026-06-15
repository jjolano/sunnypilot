# Longitudinal §3 — lead-motion anticipation at the planner↔MPC boundary

Status: implemented default-shadow/inert; validation harness hardened; apply not default-on
Date: 2026-06-14
Relates to: [hypermile tuning ADR §3](2026-06-13-longitudinal-hypermile-tuning.md),
[clean-room longitudinal architecture](2026-06-13-clean-room-longitudinal-architecture.md),
legacy ADR `docs/legacy/adr/0001-longitudinal-planner-mpc-boundary.md`, the `profile_lead_following`
drive_lab metric (`68f3c5ab12`). Spun out of the hypermile ADR as anticipated ("§3 … gets its own ADR").

## Context

The most-felt lead-following complaint is **reactive braking**: a transient dip in the radar lead's
acceleration makes the car brake more than a human would. §1 (launch) and §2 (coast margin) could not
touch this, because it is owned by the MPC, not the custom shaper — the shaper runs *after* the MPC and
can only restrict, never relax, its follow decel.

The exact boundary is `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py`:

```python
def process_lead(self, lead):
    a_lead = np.clip(lead.aLeadK, -10., 5.)      # noisy radar lead accel
    a_lead_tau = lead.aLeadTau
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)

def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)   # decayed over the horizon
    v_lead_traj = clip(v_lead + cumsum(T_DIFFS * a_lead_traj), 0, ...)
    x_lead_traj = x_lead + cumsum(T_DIFFS * v_lead_traj)
```

A momentary negative `aLeadK` spike is extrapolated into the lead's predicted trajectory → becomes the
obstacle the MPC must avoid → the MPC brakes. The lead's **velocity** is well-measured; its
**acceleration** (`aLeadK`) is the noisy term, and `aLeadTau` (~1.5 s) decays it but not fast enough for
short transients.

## Decision

Anticipate lead motion by **shaping `aLeadK` (and optionally `aLeadTau`) before `extrapolate_lead`**,
discounting the noisy accel by a **confidence** derived from lead-track stability — the same signal the
custom `lead_context.py` already computes (track age/continuity, radar corroboration, `modelProb`,
on-path score). A confident, sustained lead decel propagates at full weight; a low-confidence transient
spike is discounted and/or decayed faster, so it does not drive a reactive brake.

This is **NOT a shaper-only change.** Unlike §1/§2, it can *reduce* the braking the MPC would otherwise
command — it feeds the MPC a less-alarming lead. That makes it **safety-relevant**: if a lead really is
decelerating and we wrongly discount it, we under-brake. Hence its own ADR, a conservative confidence
model, apply-default-off, and replay validation before any apply default-on.

### Mechanism

`a_lead_shaped = a_lead * discount(confidence, a_lead)` where:

- `discount → 1.0` (no change) when confidence is high, when `a_lead >= 0` (never touch a lead that's
  *accelerating* — only the braking term is the risk), or when the decel is **sustained** across recent
  frames (a real brake, not a spike);
- `discount` floors at `DISCOUNT_MIN` (e.g. 0.5) — we never fully ignore a measured decel;
- optionally shorten `a_lead_tau` for low-confidence frames so the shaped accel decays faster.

Confidence is a function of: track continuity/age (stable, non-flickering radar track), radar
corroboration, `modelProb`, and on-path score — all already in `lead_context.LeadRelevanceState`.

### Implementation paths (decide during prototyping)

1. **SP-planner hook (preferred, stays in sunnypilot):** before `mpc.update(radarState, …)`, rewrite the
   lead's `aLeadK`/`aLeadTau` using the custom lead-confidence, on a shallow copy of the lead. One
   hook-sized call in the SP planner; no upstream-MPC edit.
2. **Touch-point in `process_lead`:** apply the discount inside the MPC lib. Smaller blast radius in code
   but edits an upstream file (record in `docs/touch-points.md`).

Either way it is gated by a mode param (default-shadow/raw passthrough; apply-default-off) and
falls back to raw `aLeadK` on any fault.

## Invariants (must not move)

- **Never raise the lead's predicted speed/position above the raw measurement** — shaping may only make
  the lead look *less* threatening via a smaller |decel|, never *more* (no faster lead → no shorter gap
  the MPC trusts). The shaped `a_lead` stays in `[a_lead, 0]` for a braking lead.
- **Never discount a high-confidence, sustained decel** — only low-confidence transients.
- `discount >= DISCOUNT_MIN`; the MPC's collision/FCW constraints, `min_x_lead`, and accel limits are
  untouched.
- Default-off; fail-closed to raw `aLeadK`.

## Validation (gate, before default-on)

- Replay the following corpus through old-vs-new lead processing (`profile_lead_following`): **reactive-
  brake share down, approach decel-peak down, steady headway NOT increased, and zero new
  close-approaches / FCW events**. The last two are the safety guardrails — anticipation must not buy
  comfort with distance.
- A closed-loop / fuzz check that a **real sustained lead brake** still produces adequate decel (the
  discount must not blunt a genuine hard brake). Reuse `fuzz_longitudinal`; mind the jerk ceiling noted
  for launch/stop.
- Widen the engaged following corpus first — the current ~3 min is too thin to trust the deltas
  (`bd`/`be` to be re-pulled cleanly), as the hypermile ADR flagged.

## Implementation + validation status (2026-06-14)

**Implemented (commit `c16fd328`, originally default-off) and validated INERT — not enabled for apply.** Built the
`LeadAnticipation` adapter + the `tools/drive_lab/replay_lead_anticipation.py` MPC A/B gate (commit
`b795484a`). The gate ran the real `LongitudinalMpc` over 6 routes / ~7000 moving-with-lead frames
(c7/b4/b5/c4/c8/c9) with raw vs §3-shaped leads: **0 softened frames, decel peak unchanged, 0 risky
softenings.** The gate **failed on benefit**, so actuated lead anticipation stayed off.

**Mode control updated (2026-06-15):** `LeadAnticipationMode=off|shadow|apply` replaces the
binary enable as the source of truth. The default is `shadow`, which warms the same tracker and
records what would have been shaped but returns the exact raw `radarState` to the MPC. `apply` is
explicit opt-in and is runtime-gated by `CustomLongitudinalEnabled`. The compatibility
`LeadAnticipationEnabled` bool is storage-only/inert; missing or empty string mode resolves to
`shadow`, never to `apply`.

Why it's inert: real radar leads are continuously-tracked → high confidence, and §3 by design never
discounts a confident lead. The confidence gate that makes it *safe* makes it do *nothing* exactly
where steady-following reactive braking happens — on a stable lead whose `aLeadK` is noisy but
"trusted." §3 only fires for a genuinely-new cut-in lead in its first ~0.45 s while braking (unit-
tested, but absent from the corpus).

**The real lever** for the felt reactive braking is temporal smoothing / a shorter `aLeadTau` applied
to **all** leads (not confidence-gated) — but that reduces braking responsiveness for confident leads
too, so it is a distinct, higher-risk change needing its own scoping + a closed-loop "still brakes for
a real decel" gate. §3 stays committed default-shadow/raw-passthrough as the foundation (the
`LeadAnticipation` adapter can host the temporal-smoothing path).

**Validation harness hardened (2026-06-14) but corpus rerun blocked in this checkout.** The replay gate
now emits machine-readable evidence fields (`benefit_detected`, `safety_pass`, `invalid_metric`) and has
synthetic tests for no-data, softened-frame counting, decel peaks, risky softenings, and JSON rendering.
This improves repeatability of the gate only; it does **not** replace route replay and does not certify
default-on safety. The historical 6-route run above remains historical evidence until rerun.

The current checkout does not include a canonical route manifest or local logs for that rerun. Existing
ADR shorthand references (`c7/b4/b5/c4/c8/c9`, `bd/be`, `000001c7`) are not valid `LogReader`
identifiers by themselves, so the corpus cannot be replayed here without full canonical route names or
local `rlog`/`qlog` files.

## Consequences

- Highest-risk longitudinal lever (can reduce braking) → most validation-gated; default-on waits on a
  clean replay with the safety guardrails intact, like the rest of the hypermile feel.
- The confidence model is reusable from `lead_context.py`; the new surface is just the lead-accel rewrite
  + its param + the replay harness extension.
- If path 2 is chosen, it is the first custom touch into the MPC lib — record it in `docs/touch-points.md`.
