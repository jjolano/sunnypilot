# Shadow-feature verdicts

Shadow modes are how behavior earns its way in (§1 launch shipped this way) or gets honestly
killed (§2/§3 refuted this way). But shadow code that never graduates is debt: it costs
maintenance, telemetry space, and reader attention. This runbook gives every currently-shadowed
longitudinal feature a decision procedure and a deadline. When a deadline passes without the
evidence to promote, the default verdict is **delete** (git history keeps the code).

Companion tooling: `tools/drive_lab/profile_shadow_heuristics.py` (scenario context + control
invariants), `profile_lead_following.py` (lead behavior gates), and the
`LongitudinalDebugTraceMode=log` route trace. The device was offline when this runbook was
written; run the harvest after the next few shadow-collecting drives.

**2026-07-24 harvest ran** — see [`../research/natural-feel-gap-analysis.md`](../research/natural-feel-gap-analysis.md)
(101,741 trace frames, routes `2cd` + `2ce`). The "Default" column below records what the
*code* defaults to; several features are set to a different mode **on the device** and have
been collecting all along. Device-observed modes are noted inline.

## Verdict table

| Feature (param) | Default | Evidence source | Decide by |
|---|---|---|---|
| Speed-aware torque (`LiveTorqueSpeedAdaptiveMode`) | off (device: **apply**) | `speed_adaptive_verdict` route replay + cross-route ratio spread | 2026-08-15 |
| Scenario context (`ScenarioContextMode`) | shadow (device: **param absent**) | `profile_shadow_heuristics` grade summary | 2026-07-20 |
| Curve traffic advisor (`CurveTrafficAdvisorMode`) | off (device: **shadow**) | debug trace `curveSpeedConfidence`/advisor fields vs SCC vision caps | 2026-07-20 |
| Lead anticipation (`LeadAnticipationMode`) | shadow (device: **param absent**) | `replay_lead_anticipation` (already run: 0 softenings) | 2026-07-20 |
| Cut-in brake assist (`CutInBrakeAssistMode`) | off (device: **shadow** — it *has* been collecting) | debug trace cut-in fields on cut-in events | 2026-08-01 |
| ~~Curve speed confidence (`CurveSpeedConfidenceMode`)~~ | **DELETED 2026-07-24** | 0.07% eligibility over 101,741 frames | resolved |
| Standstill release confidence (`StandstillReleaseConfidenceMode`) | off (device: **gate**) | stop-and-go routes, release block reasons | 2026-08-01 |
| Dynamic follow gap (`DynamicFollowGapMode`) | shadow | `profile_lead_following` A/B gate | new — collect first |
| Roll-compensation gain (`RollCompGainMode`) | off | engaged-route replay: straight-cruise tracking, crown transitions, banked curves | 2026-08-15 |
| Map coast (`MapCoastMode`) | off | debug trace `map_coast_*` fields vs manual lift-off points (`manual_longitudinal_profile`) | 2026-08-15 |

## Per-feature procedure

### Speed-aware torque
- Harvest: `uv run python -m openpilot.tools.drive_lab.speed_adaptive_verdict ROUTE [ROUTE ...]`.
- Flip `LiveTorqueSpeedAdaptiveMode` to `shadow` on-device for the collection window; the code default stays `off`.
- The analyzer forces shadow collection, replays each route through the real
  `TorqueEstimator`, fits the speed-aware profile from the collected buckets, and
  computes the per-route anchors/ratios/confidence/points plus the would-be
  `latAccelFactor` deltas.
- **Promote** (to `apply`) if >=3 routes produce confident anchors and the
  cross-route ratio spread at every confident anchor is < 0.05.
- **Park** if >=3 routes are confident but the spread is >= 0.05 (the speed
  dependency is not consistent enough to apply safely).
- **Delete** if the learner rarely meets its point/confidence thresholds in
  normal engaged driving — shadow code that never graduates is debt.

### Scenario context (grade compensation proposal)
- Harvest: `uv run python -m openpilot.tools.drive_lab.profile_shadow_heuristics ROUTE`.
- **Promote** (to a bounded apply tier) if `proposedCompensation` tracks the observed accel
  bias on sustained grades (sign agreement and magnitude within ~50% on uphill/downhill
  segments) across ≥3 routes with meaningful grade.
- **Delete** if proposals are noise or the grade classification flickers. On promotion,
  fold the three partial grade features (uphill recovery, downhill overspeed leeway, this
  compensation) into one grade-aware coast model around `coast_horizon.coast_decel_from_grade`
  + `DragEstimator`; on deletion, keep the two shipped behaviors as-is.

### Curve traffic advisor
- Compare its proposed caps against what actually bound (SCC vision cap, now runway-governed
  in `policy._advisory_curve_cap`). The advisor only earns promotion if it would have
  prevented a late-braking curve entry or a driver intervention that the governed vision cap
  missed — check curve segments with brake/steer overrides.
- **Delete** if the runway-governed vision cap covers its value; it overlaps heavily now.

### Lead anticipation (§3)
- Already validated inert on 6 routes / ~7000 frames (0 softenings; following braking is
  genuine closing, not aLeadK noise). The dynamic follow gap is the honest replacement lever.
- **Verdict: retire the apply path** once `DynamicFollowGapMode` has its first validated
  apply data — keep only if that replay shows anticipation adding measurable softening on top
  of the compressed gap (it did not on the fixed gap).

### Cut-in brake assist / curve speed confidence / standstill release confidence
- They default **off** in code, but the device has run all three in a collecting mode
  (`shadow`/`shadow`/`gate`). The 2026-07-24 harvest over 101,741 frames gives:
  - **Cut-in brake assist** — eligible on **84 frames (0.08%)**, and every one came from
    route `2cd`; `2ce` contributed zero. When eligible the numbers are coherent (TTC median
    4.12 s, required decel 0.70, proposed cap **−0.99 m/s²**, confidence 1.00) and the
    proposed cap sits between the old OP cut-in peak (−2.0) and the human baseline (−0.47).
    **Verdict: hold in shadow.** Directionally right, far too sparse to promote. Needs a
    cut-in-rich engaged corpus; note OP produced 0 valid cut-in brake reactions in 29.4
    engaged minutes, so the underlying harshness claim is itself untested on this build.
  - **Curve speed confidence** — eligible on **75 frames (0.07%)**, active 0.14%.
    **Verdict: deleted 2026-07-24.** Exactly the "rarely meets its thresholds" case. Removed:
    the predictor, `CurveSpeedConfidenceMode` (param key, sunnylink YAML/JSON, UI schema), the
    finalizer's `scc_curve_confidence_final_cap` and its three constants, and the planner trace
    populate. Kept: `CurveSpeedConfidenceInputs` — it is the shared SCC curve-evidence carrier
    that the curve traffic advisor reads — and the `curveSpeedConfidence` capnp ordinal @20,
    marked retired so old logs still decode.
  - **Standstill release confidence** — running as `gate`; eligible 0.74%, 95.6% blocked on
    `no_release_permission`. Working as intended; no change.

### Dynamic follow gap (new)
- Shadow first: confirm `eligible` fires on real approaches and the would-be `t_follow`
  trace looks sane (compress on approach, fast recovery on lead braking).
- Apply gate (before enabling `apply` + research actuation): `profile_lead_following` on
  matched routes must show approach decel peak **down**, zero new close approaches
  (min time gap ≥ 1.05 s), and headway recovery after the approach. This is a deliberate
  headway tradeoff — it only ships with that replay evidence.

### Map coast (new)
- Coast-only lift-off toward SCC-Map slowdowns beyond vision range: `coast_v_target`/`coast_distance`
  from the map controller's coast pass (600 m lookahead, same route-hygiene gates as braking) feed
  `coast_horizon` as an `ADVISORY_CAP` floored at the natural coast decel — map evidence never brakes.
  Apply is gated by `CustomLongitudinalEnabled` + `AllowLongitudinalResearchActuation` + the
  SmartCruiseControlMap toggle (CURVE_MAP admissibility) + SCC mode.
- Shadow first: flip `MapCoastMode=shadow` with `LongitudinalDebugTraceMode=log`, drive mapped roads,
  compare `map_coast_cap`/`map_coast_eligible` lift-off points against manual lift-off from
  `manual_longitudinal_profile` on the same approaches.
- **Promote** (to apply) if shadow lift-off points land within ~2 s of the manual baseline's on ≥3
  routes with mapped slowdowns and there are no false targets (eligible firing with no real
  slowdown ahead) — stale OSM data is the failure mode to catch.
- **Delete** if OSM targets are too sparse/stale on the actual driven routes for the tier to fire,
  or false targets appear that the route-hygiene gates don't catch.

### Roll-compensation gain
- Learning is already shadow-capable; this runbook covers the apply gate now that the live
  path exists. Flip `RollCompGainMode` to `apply` only after replay evidence exists.
- Harvest: engaged-route replay comparing fixed `ROLL_COMPENSATION_GAIN` (0.55) versus the
  learned gain on routes with sustained crown/cross-slope and gentle banked curves.
- **Promote** (to a recommended apply default) if this single fleet's replay on ≥3 routes
  shows straight-cruise tracking error no worse than fixed 0.55, each route's
  `roll_comp_profile` `roll_span` is ≥ 0.3 m/s², and the learned gain spread across those
  routes is < 0.05.
- **Delete the apply path and keep shadow** if the fixed constant is indistinguishable or more
  consistent; keep the learner as observability only.
- **Delete the feature entirely** if shadow telemetry shows the learned gain is noisy or the
  fit span/confidence thresholds are rarely met in normal driving.
- This gate is intentionally single-fleet and route-based; per-platform generalization is
  deferred until the fleet grows.
