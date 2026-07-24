# Natural-feel gap analysis: manual vs engaged on the current build

**Date:** 2026-07-24
**Corpus:** routes `000002c8`–`000002d1` (2026-07-22 → 2026-07-24)
**Build under test:** the build preceding `5d86dc4a4a` — see "Build caveat" below
**Purpose:** Re-measure where engaged driving differs from the driver's own driving, on the
current stack, so that comfort work is aimed rather than remembered. Measurement only; no
product code was changed.

Prior baselines this supersedes in part:
[`lead-reaction-baseline-analysis.md`](lead-reaction-baseline-analysis.md) (2026-06-21,
commit `16c448350`) and [`manual-lateral-baseline-analysis.md`](manual-lateral-baseline-analysis.md).

---

## Build caveat — read first

Today's commit `5d86dc4a4a` was authored 16:56. Routes `2d0` (18:20) and `2d1` (19:10) ran
after it, but **both were driven entirely manually**. The current build therefore has **zero
engaged miles**. Every engaged number below comes from `2cd` and `2ce`, which ran on the
previous build. Nothing here validates today's commit.

## Engagement triage

Engagement was checked before any analysis, because this device regularly logs full drives
with cruise main never on.

| Route | segs | longActive | latActive | role |
|---|---:|---:|---:|---|
| `2c8` | 24 | 0.0% | 0.0% | manual reference |
| `2c9` | 29 | 0.0% | 0.0% | manual reference |
| `2ca` | 40 | 15.1% | 62.8% | mixed |
| `2cb` | 23 | 0.0% | 0.0% | manual reference |
| `2cc` | 29 | 0.0% | 0.0% | manual reference |
| **`2cd`** | 46 | **50.2%** | **79.7%** | primary engaged |
| `2ce` | 40 | 16.7% | 83.1% | engaged |
| `2cf` | 25 | 0.0% | 0.0% | manual reference |
| `2d0` | 15 | 0.0% | 0.0% | manual reference |
| `2d1` | 18 | 0.0% | 0.0% | manual reference |

**7 of 10 routes had zero engagement.** Total engaged-with-lead exposure across the whole
corpus is 29.4 minutes. This is the binding constraint on every longitudinal result below.

rlogs analyzed: `2cd` (46 segs), `2ce` (40), `2d1` (18, manual), plus partial `2cf`/`2ca`.
`latActive` far exceeds `longActive` on every mixed route, so lateral evidence is roughly
6× richer than longitudinal.

---

## Finding 1 — Steady-cruise chatter is in the *response*, not the command

`profile_cruise_smoothness` (run here for the first time against real routes), `a_ego` — the
only acceleration channel both modes share:

| | manual `2cd` (n=8) | manual `2d1` (n=17) | engaged `2cd` (n=197) |
|---|---:|---:|---:|
| accel stddev | 0.152 | 0.116 | **0.106** |
| jerk p90 | 2.79 | 3.57 | **4.19** |
| jerk p99 | 6.18 | 7.84 | **10.26** |
| sign reversals / min | 35.4 | 78.3 | **150.0** |
| deadband share | 0.137 | 0.241 | 0.411 |

Engaged holds speed *tighter* (lowest stddev) but with ~1.3× the jerk p90 and ~2.3× the
sign reversals of pooled manual.

**The commanded channels are flat.** In the same engaged windows, `car_control_accel` shows
deadband share **1.000**, **0.0** reversals/min and jerk p99 **0.085 m/s³**;
`longitudinal_plan_a_target` likewise. Channel coverage was 100% on all channels, so this is
not a sampling artifact.

So the steady-cruise chatter is **not** policy sign-chatter — the planner output is nearly
constant. It appears between command and measured acceleration, which points at powertrain
response (Toyota PCM gas unwind / torque-converter behaviour) rather than at any custom
longitudinal layer. **Do not spend tuning effort on the policy for this.**

Caveat: manual n (25 windows) is far smaller than engaged (197), and the steady-cruise
filter selects different road stretches per mode.

## Finding 2 — Lateral is smoother than the driver on average, with a fatter tail in corners

`manual_lateral_baseline`, `2cd` + `2d1`, 248,863 accepted samples (manual 63,366 / engaged
185,497).

By speed band, engaged wins everywhere — lateral jerk p95:

| speed | manual | engaged |
|---|---:|---:|
| 3–8 m/s | 1.355 | **0.788** |
| 8–12 m/s | 0.955 | **0.621** |
| 12–18 m/s | 0.928 | **0.658** |
| 18–30 m/s | 1.188 | **0.694** |

By cornering load, it inverts — lateral jerk p95:

| lat accel | manual | engaged | |
|---|---:|---:|---|
| 0.0–0.3 m/s² | 0.805 | 0.609 | engaged smoother |
| 0.3–0.8 m/s² | 1.846 | 2.064 | engaged rougher |
| **0.8–1.5 m/s²** | **1.798** | **2.504** | engaged **1.39×** rougher |

Tracking error rises with the same variable (engaged `err_p95` 0.100 → 0.220 → 0.391).

Clean corner-exit unwind, commanded lateral-accel jerk, engaged samples filtered to
`latActive ∧ ¬steeringPressed ∧ |driver torque| < 20`:

| | n | p50 | p90 | p95 | **p99** |
|---|---:|---:|---:|---:|---:|
| engaged (clean) | 3,077 | 0.24 | 0.74 | 1.18 | **3.17** |
| manual | 4,625 | 0.30 | 1.47 | 1.79 | **2.55** |

**The defect is a tail, not a mean.** OP unwinds corners more smoothly than the driver at
p50–p95 and only loses at p99. Any fix should target excursion suppression in the cornering
band, not global smoothing — global smoothing would make the p50–p95 advantage worse for no
gain where it actually hurts.

`lateral_comfort_imu` over `2cd` (32.2 min engaged-at-speed) agrees and localizes it:
route-level measured jerk p95 **0.78**, p99 **1.55 m/s³** — far under UN R79's 5.0 limit.
Only **16 of 114** worst events are control-attributed, and they concentrate by speed:

| regime | events | control-attributed |
|---|---:|---:|
| city (<12 m/s) | 23 | 9 (**39%**) |
| highway (≥12 m/s) | 91 | 7 (8%) |

### Hand-verification of the worst event

Per the plan's verification requirement, the single worst control-attributed event was
dissected: `2cd` segment 3, t = 226.0 s, measured +4.77 m/s³, commanded 3.91, corr 0.71.

It is a corner exit — wheel unwinds 108° → 3° over ~2 s while accelerating 8.6 → 11.9 m/s.
Commanded curvature runs −0.0379 → −0.0095 in 0.7 s, i.e. commanded lateral accel 3.28 →
0.88 m/s² ≈ **3.4 m/s³**, consistent with the reported commanded jerk.

**This event is contaminated and does not on its own prove an OP-owned defect:** driver
torque was 100–250 Nm through the exit and `steeringPressed` flickers True/False across the
window. The clean-corner table above is the trustworthy version of the same question.

### Attribution of the corner tail: model-led, and the built mitigation cannot reach it

Follow-up pass over `2cd` + `2ce`, 7,732 clean engaged corner samples, 48 excursions
(|commanded jerk| > 3.0 m/s³, 0.62% of corner samples). Median |jerk| by pipeline stage
during those excursions, against the corner-sample baseline:

| stage | during excursions | baseline |
|---|---:|---:|
| raw model curvature | **3.30** | 0.37 |
| conditioned demand | 3.48 | — |
| processed demand | 3.64 | — |
| commanded curvature | 3.58 | 0.29 |

The raw model curvature already carries a ~9× jerk spike; the pipeline adds only ~8% on top.
**The excursions are model-led, not pipeline-introduced** — the same conclusion previously
reached for the 0.15 Hz wander.

Flag rates during excursions vs baseline corner samples:

| flag | excursion (n=48) | baseline (n=7,732) |
|---|---:|---:|
| `gated` | **64.6%** | 0.9% |
| `quality` (median) | **0.65** | 1.00 |
| `previewAssistApplied` | 14.6% | 66.4% |
| `laneCenteringActive` | 16.7% | 53.1% |
| `spsApplied` / `laneRateDamping` / `laneFitSource` | 0% | 0% |

Gate `reason` during excursions: `low_lane_confidence` 33/48, `ok` 10/48, `high_path_std` 4/48.

So in a corner the model path degrades, the quality gate fires, and **both assist layers drop
out simultaneously** (preview 66% → 15%, LCA 53% → 17%) — smoothing disappears exactly when
the demand is jumpiest.

**The built mitigation is scoped out of this regime.** `model_path_processor.py` contains a
demand-jerk-smoothing stage (`DEMAND_JERK_SMOOTH_*`, caps `[1.0, 1.4, 1.8]` m/s³ over
8–22 m/s) that ADR 2026-07-03 describes as aimed at "the residual raw model-path jump
family" — exactly this. It is default-off (`demand_jerk_smoothing_enabled = False`, no param
key). Tracing its gates against the 48 measured excursions:

- `_demand_jerk_smoothing_gates_ok` requires quality ≥ **0.85** when
  `reason == "low_lane_confidence"`; excursion quality median is **0.65** → fails.
- `_demand_jerk_smoothing_eligible` then has an explicit
  `if reason == "low_lane_confidence": return False` unless the sample is *near-straight*
  (|lat accel| ≤ 0.35); excursions sit at 0.5–2.5 m/s² → fails.
- `reason == "high_path_std"` (4/48) falls through to `else: return False`.
- `turn_curvature_sign != 0` is excluded outright.

**Conclusion: enabling the existing smoother would fire on approximately none of the measured
excursions.** Reaching them requires widening gates that were deliberately drawn to avoid
smoothing demand while the path is untrusted — a safety-relevant scope change, not a flag flip.

## Finding 3 — Lead-reaction gaps: one closed, one unresolved, one untestable

`profile_lead_reaction` pooled over `2cd` + `2ce` (29.4 min OP engaged, 42.8 min manual moving):

| metric | OP | manual | n (OP/manual) | verdict |
|---|---:|---:|---:|---|
| lead speed-change reaction | 0.991 s | 0.621 s | **4** / 8 | **unresolved — n too small** |
| lead-exit accel reaction | 0.968 s | 0.877 s | 135 / 677 | **near parity — gap closed** |
| cut-in brake reaction | — | 0.983 s | **0** / 1 | **untestable on this corpus** |

- The lead-exit result is solid (n=135) and shows OP essentially matching the driver. The
  2026-06-21 doc already called this one comparable; it still is.
- The lead speed-change gap *appears* to persist and even widen (+59% vs the old +41%), but
  **n=4 is not publishable**. Do not cite this number as evidence for or against any change.
- OP produced **zero valid cut-in brake reactions** across 30 detected cut-in candidates.
  The 2026-06-21 headline finding — OP cut-in peak decel −2.000 vs manual −0.470, "4.3×
  harsher" — is therefore **neither confirmed nor refuted here**. It remains an open claim
  on an old build.

## Finding 4 — Shadow-feature harvest (deadlines overdue)

`longitudinalPlanSP.longitudinalDebug`, 101,741 trace-enabled frames over `2cd` + `2ce`.
Deadlines in [`../plans/2026-07-02-shadow-feature-verdicts.md`](../plans/2026-07-02-shadow-feature-verdicts.md)
passed on 2026-07-20.

| feature | mode | firing rate | verdict |
|---|---|---:|---|
| `curveSpeedConfidence` | shadow | eligible **0.07%** (75 frames), active 0.14% | **delete** — meets the runbook's "rarely meets its thresholds" rule |
| `cutInBrakeAssist` | shadow | eligible **0.08%** (84 frames, **all from `2cd`**; `2ce` contributed zero) | **hold in shadow** — sane but far too sparse to promote |
| `standstillReleaseConfidence` | gate | eligible 0.74%; 95.6% blocked `no_release_permission` | active as a gate; no change |
| `accEnvelope` | advisory | `wouldCap` **11.1%** | see note below |
| `leadPathClearance` | — | trace never populated | investigate or remove the trace |

When `cutInBrakeAssist` is eligible its numbers are coherent: TTC median 4.12 s, required
decel 0.70, **proposed cap −0.99 m/s²**, confidence 1.00. That proposed cap sits between the
old OP cut-in peak (−2.0) and the human baseline (−0.47) — directionally right, but 84
frames from a single route cannot carry a promotion.

Note on `accEnvelope`: the 11.1% `wouldCap` rate looks alarming but is concentrated at very
low speed (`vEgo` median **1.3 m/s**, p10 ≈ 0) — creep and stop-and-go manoeuvring, not
cruise. Of would-cap frames, 38.7% exceed the envelope by >0.2 m/s². Treat as a low-speed
signal, not a cruise-comfort one.

Note on the runbook itself: it records `CutInBrakeAssistMode` as "not even collecting". That
is wrong — the device has had it in `shadow` throughout. The verdict table needs correcting.

---

## Ranked conclusions

1. **Engagement is the binding constraint, not analysis capability.** 7 of 10 routes had zero
   engagement; the entire corpus yields 29.4 engaged minutes. Three of four longitudinal
   questions above are n-limited, and today's build has no engaged miles at all. The
   highest-leverage action for answering "does it feel natural" is **more engaged driving**,
   particularly in lead-rich traffic — not more tooling.
2. **The lateral tail in corners is the best-evidenced real defect** (engaged jerk p95 2.504
   vs manual 1.798 in the 0.8–1.5 m/s² band; p99 unwind 3.17 vs 2.55; 39% of city worst
   events control-attributed). It is a tail problem — target excursions, not the mean.
   Attribution says it is **model-led** (raw jerk 3.30 vs 0.37 baseline, pipeline adds ~8%),
   coincident with the quality gate firing on `low_lane_confidence` and both assist layers
   dropping out. The existing demand-jerk smoother is gated out of exactly this regime, so
   there is **no flag-flip fix**; the options are (a) widen the smoother's gates into
   untrusted-path corners, (b) ramp the assist drop-out instead of stepping it, or
   (c) accept it as a model floor, as was done for the 0.15 Hz wander.
3. **Steady-cruise chatter is not a policy defect.** Commands are flat; the roughness is in
   the powertrain response. Closing this would be a car-interface problem, not a tuning one.
4. **`curveSpeedConfidence` should be deleted** — 0.07% eligibility over 101,741 frames.
5. **The cut-in harshness claim is stale and untested.** It needs a cut-in-rich engaged
   corpus before it drives any change.

## Method notes / limitations

- Manual and engaged samples come from different road stretches; window-selection filters
  differ per mode. Treat all manual-vs-engaged deltas as descriptive, not causal.
- `lateral_comfort_imu` masks on `latActive ∧ ¬steeringPressed`, so its events can still sit
  adjacent to driver-torque episodes — as the hand-verified event shows.
- Manual steady-cruise window count (25) is small relative to engaged (197).
- Reproduce with the throwaway roll-up scripts in this session's scratchpad; they were
  deliberately not promoted into `tools/drive_lab/` (no route-level comfort score exists, and
  inventing one was out of scope).
