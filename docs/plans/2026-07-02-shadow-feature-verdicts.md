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

## Verdict table

| Feature (param) | Default | Evidence source | Decide by |
|---|---|---|---|
| Scenario context (`ScenarioContextMode`) | shadow | `profile_shadow_heuristics` grade summary | 2026-07-20 |
| Curve traffic advisor (`CurveTrafficAdvisorMode`) | off | debug trace `curveSpeedConfidence`/advisor fields vs SCC vision caps | 2026-07-20 |
| Lead anticipation (`LeadAnticipationMode`) | shadow | `replay_lead_anticipation` (already run: 0 softenings) | 2026-07-20 |
| Cut-in brake assist (`CutInBrakeAssistMode`) | off | debug trace cut-in fields on cut-in events | 2026-08-01 |
| Curve speed confidence (`CurveSpeedConfidenceMode`) | off | debug trace + curve routes | 2026-08-01 |
| Standstill release confidence (`StandstillReleaseConfidenceMode`) | off | stop-and-go routes, release block reasons | 2026-08-01 |
| Dynamic follow gap (`DynamicFollowGapMode`) | shadow | `profile_lead_following` A/B gate | new — collect first |

## Per-feature procedure

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
- All default **off** — they are not even collecting. Either flip them to shadow on-device
  for the collection window, or accept the delete-by-default deadline. A shadow feature that
  nobody turns on answers its own question.

### Dynamic follow gap (new)
- Shadow first: confirm `eligible` fires on real approaches and the would-be `t_follow`
  trace looks sane (compress on approach, fast recovery on lead braking).
- Apply gate (before enabling `apply` + research actuation): `profile_lead_following` on
  matched routes must show approach decel peak **down**, zero new close approaches
  (min time gap ≥ 1.05 s), and headway recovery after the approach. This is a deliberate
  headway tradeoff — it only ships with that replay evidence.
