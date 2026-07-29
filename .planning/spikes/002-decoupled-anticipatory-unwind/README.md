---
spike: 002
name: decoupled-anticipatory-unwind
type: comparison
validates: "Given Spike 001's unsafe feedback and weak unwind result, when fallback state is kept unshaped and the same jerk shaper optionally follows a causal 100-200 ms unwind preview, then the lag cap remains strict and unwind jerk falls without new excursions"
verdict: PARTIAL
related: [001]
tags: [lateral, jerk, corner, replay, preview]
---

# Spike 002: Decoupled anticipatory unwind

## What This Validates

Can a separate unshaped fallback reference make post-gate demand shaping safe, and can
the model's existing causal path preview improve unwind without retaining stale steering
demand?

## Research

| Approach | Reuses | Benefit | Problem | Status |
|---|---|---|---|---|
| Decouple fallback reference only | Spike 001 post-gate shaper | Removes shaped-state feedback | Does not anticipate unwind | Validated offline |
| Aim the same shaper at 100/200 ms preview | `PreviewAssistTracker._preview_curvature` | Causal early unwind, no new stage | Preview offset can collide with a moving reference cap | Invalidated |
| Soft-release preview on transient gates | Existing preview-assist release pattern | Tests gate drop-out as the failure source | Did not change any replay metric | Invalidated |

No external research was needed. The experiment reuses the repository's model-path
processor, curvature-from-plan helper, replay extraction, and cached rlogs.

## How to Run

From a clean checkout of commit `e2bc853515`:

```bash
SUNNYPILOT_REPO_ROOT="$PWD" uv run --extra testing --extra tools python \
  /path/to/.planning/spikes/002-decoupled-anticipatory-unwind/experiment.py \
  /tmp/opencode/sunnypilot-route-logs/000002cd--528f8f8262 \
  /tmp/opencode/sunnypilot-route-logs/000002ce--7be8aa5a07 \
  --output /path/to/.planning/spikes/002-decoupled-anticipatory-unwind/results.json
```

Fast logic check:

```bash
SUNNYPILOT_REPO_ROOT="$PWD" uv run --extra testing --extra tools python \
  /path/to/.planning/spikes/002-decoupled-anticipatory-unwind/experiment.py --self-check
```

## What to Expect

The console and `results.json` compare strict lag, equivalent delay, turn-in/unwind jerk
tails, new excursions, and sign mismatches for decoupled and anticipatory variants.

## Observability

`results.json` preserves route/segment coverage, per-phase distributions, safety counters,
and the largest improved events.

## Investigation Trail

1. Spike 001 showed that post-gate shaping helped entry but exceeded its nominal lag cap
   because shaped controller demand became the next fallback reference.
2. This spike gives `ModelPathProcessor` its own unshaped prior result. The same existing
   slew/clamp operation shapes only the returned controller demand.
3. Both decoupled variants retained Spike 001's jerk reduction while hitting their caps
   exactly, with zero new excursions and zero sign mismatches.
4. A causal unwind target reused the model orientation path and
   `get_curvature_from_plan`. The 100 ms and 200 ms variants produced identical output:
   that helper linearizes every action time below `MIN_STABLE_DELAY = 0.3 s`.
5. Preview reduced unwind p99 slightly further, but introduced 10 new excursions. A
   follow-up soft-release variant produced identical metrics, so transient preview gate
   drop-out is not a sufficient explanation.
6. The unchanged release result is consistent with an anticipatory offset meeting a
   moving unshaped reference and forcing the strict cap to catch up. The modest unwind
   gain does not justify another preview variant, so the branch stops here.

## Results

**Verdict: PARTIAL.** Replaying commit `e2bc853515` over all 86 cached rlog segments from
`2cd` and `2ce` produced 509,379 frames and 22,838 clean corner frames. The filter found
30 baseline excursions above `3.0 m/s³`.

| Variant | >3 excursions | Turn-in p99 | Unwind p99 | Max lag | Equivalent delay | New excursions | Safety |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 30 | 2.234 | 2.913 | — | — | — | — |
| Decoupled, 0.08 | 20 | 1.779 | 2.820 | 0.080 | 78 ms | 0 | PASS |
| Decoupled, 0.20 | **17** | **1.568** | 2.820 | 0.200 | 191 ms | 0 | **PASS** |
| Preview 100 ms, 0.20 | 22 | 1.580 | **2.752** | 0.200 | 179 ms | **10** | FAIL |
| Preview 200 ms, 0.20 | 22 | 1.580 | **2.752** | 0.200 | 179 ms | **10** | FAIL |
| Preview 200 ms + soft release | 22 | 1.580 | **2.752** | 0.200 | 179 ms | **10** | FAIL |

**Build signal:** preserve a separate unshaped fallback reference if post-gate shaping is
implemented. Do not add path-preview targeting to this shaper. Decoupling validates safe
turn-in improvement, but unwind needs a different owner or better reference signal.

Limitations: this remains fixed-DT open-loop pipeline replay, not closed-loop controller or
vehicle proof.
