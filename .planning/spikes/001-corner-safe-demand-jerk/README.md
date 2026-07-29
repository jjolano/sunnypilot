---
spike: 001
name: corner-safe-demand-jerk
type: comparison
validates: "Given clean engaged corner transitions from routes 2cd/2ce, when the existing demand slew is tested before and after low-quality fallback with 0.08 and 0.20 m/s² lag caps, then turn-in/unwind jerk tails shrink without sign error or more than 200 ms equivalent lag"
verdict: PARTIAL
related: []
tags: [lateral, jerk, corner, replay]
---

# Spike 001: Corner-safe demand jerk

## What This Validates

Can the existing demand-jerk rate limiter cover loaded corners without creating a second
governor or retaining unsafe steering demand during unwind?

## Research

| Approach | Reuses | Benefit | Problem | Status |
|---|---|---|---|---|
| Widen the existing pre-gate smoother | `ModelPathProcessor` rate/lag clamp | Smallest production-shaped change | Low-quality frames return before it runs | Tested, ineffective on excursions |
| Move that operation after low-quality fallback | Same rate/lag clamp | Reaches the measured regime | Feeds shaped state back into later fallback decisions | Tested, promising but unsafe as written |
| Add a final lateral-demand governor | New stage | Easy global bound | Duplicates output-governor authority and obscures attribution | Rejected |
| Hold the last trusted curve | Retired CurveMemory idea | Can bridge perception loss | Capture was gate-starved and crawl turns lacked trustworthy history | Rejected |

The comparison keeps the existing speed-resolved jerk schedule and varies placement plus
the lateral-acceleration lag cap:

- `lag08`: current `0.08 m/s²` cap, roughly 40–80 ms at the existing jerk schedule.
- `lag20`: `0.20 m/s²` cap, the user's accepted 100–200 ms softness budget.

No external library research is needed; this is pure replay logic using the repository's
existing `LogReader`, route extraction, demand pipeline, and smoother clamp.

## How to Run

```bash
uv run --extra testing --extra tools python \
  .planning/spikes/001-corner-safe-demand-jerk/experiment.py \
  /tmp/opencode/sunnypilot-route-logs/000002cd--528f8f8262 \
  /tmp/opencode/sunnypilot-route-logs/000002ce--7be8aa5a07 \
  --output .planning/spikes/001-corner-safe-demand-jerk/results.json
```

Fast logic check:

```bash
uv run --extra testing --extra tools python \
  .planning/spikes/001-corner-safe-demand-jerk/experiment.py --self-check
```

## What to Expect

The JSON and console summary report turn-in, unwind, and all-corner jerk tails; excursion
counts above `3.0 m/s³`; new excursions introduced by shaping; maximum lag; equivalent
delay; sign mismatches; and the worst changed events.

## Observability

`results.json` records route/segment coverage, gate counts, per-variant metrics, and the
largest baseline-to-candidate event changes with timestamps.

## Investigation Trail

1. Existing evidence localizes the defect to model demand during loaded corners:
   48 clean excursions on `2cd`/`2ce`, mostly while `low_lane_confidence` gates are active.
2. The built demand smoother is already rate- and lag-bounded, but its near-straight,
   quality, and reason gates exclude approximately all of those excursions.
3. CurveMemory is not a fallback candidate: it was deleted after capturing only 2 of 1,912
   eligible fast frames and cannot recover turns where vision was never trustworthy.
4. The experiment therefore changes only offline eligibility and lag allowance. It leaves
   production code untouched.
5. First replay showed why simple eligibility widening misses the defect: quality below
   `0.75` returns at step 6, while demand-jerk smoothing runs at step 8.
6. A follow-up variant reused the same operation after the low-quality fallback. This
   reached the target events, but feeding the shaped result back as
   `previous_desired_curvature` let total baseline-relative lag drift past the nominal cap.

## Results

**Verdict: PARTIAL.** Replaying commit `e2bc853515` over all 86 cached rlog segments from
`2cd` and `2ce` produced 509,379 frames and 22,838 clean corner frames. The spike's filter
found 30 baseline excursions above `3.0 m/s³`.

| Variant | >3 excursions | Turn-in p99 | Unwind p99 | Max lag | Equivalent delay | Safety |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 30 | 2.234 | 2.913 | — | — | — |
| Pre-gate, 0.08 | 30 | 2.073 | 2.913 | 0.080 | 69 ms | PASS |
| Pre-gate, 0.20 | 30 | 1.908 | 2.913 | 0.200 | 163 ms | PASS |
| Post-gate, 0.08 | 20 | 1.779 | 2.820 | **0.084** | 78 ms | FAIL |
| Post-gate, 0.20 | **17** | **1.568** | 2.820 | **0.211** | 191 ms | FAIL |

The post-gate `0.20` variant removed 43% of excursions and cut turn-in p99 by 30%, with
zero new excursions and zero material sign mismatches. It did not solve unwind: p99 fell
only 3%. Both post-gate variants failed the strict lag cap because shaped output feeds back
as the next frame's fallback reference; total divergence reached 0.084 vs 0.08 and 0.211
vs 0.20 m/s².

**Build signal:** do not widen the current gate or ship the post-gate wrapper. A follow-up
must separate the unshaped fallback reference from the shaped controller demand and address
unwind anticipatorily rather than merely retaining old-direction demand.

Limitations: fixed-DT open-loop pipeline replay is not vehicle/controller closed-loop proof,
and this spike's filters differ from the prior 48-excursion research roll-up.
