# Manual-driving baseline analysis

**Date:** 2026-06-17
**Author:** AI-assisted analysis driven by user's manual-driving logs
**Data source:** Personal comma device, qlogs from daily driving
**Routes:** 15 routes, 441 segments, ~6.3M messages
**Purpose:** Tune the `lead_speed_alignment` decel ramp and TTC gates to match human
behavior for the custom-2.0 longitudinal policy.

---

## Methodology

### Data collection

5 raw qlog.zst segments per route were pulled from the device over Tailscale
(`comma@100.94.10.12`). Each segment contains carState, carControl, radarState,
and longitudinalPlan messages sampled at ~7 Hz (qlog decimation).

### Sample filters

All samples within each segment were evaluated:

| Filter | Rationale |
|---|---|
| `carControl.longActive == False` | Manual driving only; skip ACC-engaged periods |
| `radarState.leadOne.status == True` | A primary lead must be present |
| `carState.vEgo >= 8.0 m/s` | Highway speed; exclude stop-and-go |
| `leadOne.dRel` in (0, 250] m | Physically plausible lead distance |
| `v_ego + v_rel` in [-2, 45] m/s | Plausible lead absolute velocity |

16,494 samples passed all filters.

### Action labeling: aEgo-based (not pedal switches)

Earlier analysis used pedal switches (`brakePressed`, `gasPressed`) to label driver
action. This was found to systematically **under-count** deceleration:

- `brakePressed` only fires when the brake pedal switch closes — engine braking,
  regen, and trailing throttle all produce deceleration without it.
- `gasPressed` can stay true while foot rests on the pedal during coast-down.

Corrected labels use **actual vehicle deceleration** (`carState.aEgo`):

| aEgo range | Label |
|---|---|
| `< -1.5 m/s²` | **hard_brake** |
| `-1.5 to -0.6` | **med_brake** |
| `-0.6 to -0.3` | **light_brake** |
| `-0.3 to +0.3` | **coast** |
| `> +0.3` | **accel** |

Impact: In the critical 0.35–0.80 required-decel band, pedal labels reported
~38% brake — aEgo labels show **62–75% brake**.

### Feature computation

Each sample computes:

```
closing       = max(0, -vRel)
desired_gap   = max(follow_gap, 1.5 * v_ego)
excess_gap    = dRel - desired_gap
required_decel = (closing^2) / (2 * excess_gap)   [if excess > 0.1 and closing > 0.05]
ttc           = dRel / closing                     [if closing > 0.05]
thw           = dRel / v_ego
```

### Closing events

Only the subset with `closing > 0.05 m/s` and `excess_gap >= 3.0 m` is used for
decel-band analysis (7,227 events, 5,246 after the excess-gap filter — the
difference is `excess_gap < 3.0` events that are too close to the desired gap
to be meaningful for alignment).

---

## Core tables

### Table 1: TTC crossover (closing events, aEgo labels)

```
TTC bucket |    n | brake% | coast% | accel%
---------- | ---- | ------ | ------ | ------
<2         |    1 | 100.0% |   0.0% |   0.0%
 2-4       |  222 |  99.1% |   0.9% |   0.0%
 4-6       |  418 |  90.9% |   8.9% |   0.2%
 6-8       |  341 |  79.2% |  19.6% |   1.2%
 8-10      |  289 |  70.9% |  27.0% |   2.1%
10-12      |  224 |  64.3% |  32.1% |   3.6%
12-14      |  263 |  57.0% |  38.8% |   4.2%
14-16      |  182 |  58.8% |  39.0% |   2.2%
16-18      |  196 |  45.4% |  51.0% |   3.6%
18-20      |  149 |  33.6% |  58.4% |   8.1%
20-22      |  165 |  30.9% |  61.8% |   7.3%
22-24      |  116 |  36.2% |  57.8% |   6.0%
>24        | 2680 |  20.5% |  68.0% |  11.5%
```

Key inflection points:
- **90%+ braking:** TTC < 6 s
- **Brake dominant:** TTC < 16 s
- **Coast/brake crossover:** TTC ≈ 16–18 s
- **Comfort/progress zone:** TTC > 24 s (only 20% brake)

### Table 2: Required decel bands (closing events, 1.5s gap, aEgo labels)

```
reqDecel    |    n | brake% | coast% | H%  | M%  | L%  | medTTC | medvEgo
----------- | ---- | ------ | ------ | --- | --- | --- | ------ | -------
 0.00-0.05  | 2013 |  18.3% | 68.0%  | 0.1 | 1.9 |16.2 |  83.8s |  16.1
 0.05-0.10  |  624 |  31.7% | 62.7%  | 1.8 | 8.5 |21.5 |  29.0s |  15.9
 0.10-0.15  |  355 |  34.4% | 62.3%  | 2.8 | 7.6 |23.9 |  21.8s |  16.1
 0.15-0.20  |  252 |  34.9% | 57.5%  | 2.8 | 7.1 |25.0 |  18.3s |  16.8
 0.20-0.25  |  182 |  50.5% | 42.9%  | 2.7 |18.1 |29.7 |  15.9s |  14.9
 0.25-0.30  |  159 |  57.9% | 39.0%  | 3.1 |24.5 |30.2 |  13.9s |  15.6
 0.30-0.35  |  146 |  54.1% | 39.7%  | 2.1 |20.5 |31.5 |  13.0s |  15.3
 0.35-0.50  |  224 |  62.1% | 35.7%  | 3.6 |16.5 |42.0 |  11.4s |  14.4
 0.50-0.80  |  349 |  75.1% | 23.8%  | 6.0 |28.7 |40.4 |   8.9s |  13.5
 0.80-2.00  |  622 |  82.0% | 17.4%  |19.9 |31.2 |30.9 |   6.0s |  12.3
 >2.00      |  320 |  96.2% |  3.8%  |45.3 |42.8 | 8.1 |   3.7s |  11.1
```

(H=hard_brake, M=med_brake, L=light_brake — see aEgo thresholds above.)

Key insight: **Humans brake across ALL bands.** The 0.35–0.80 former silent gap
has 62–75% human braking (mostly light to medium). Filling this gap with a
capped advisory was the top priority.

### Table 3: Speed × required decel (1.5s gap, aEgo labels)

#### >=0.80 reqDecel across speed bands

```
Speed      |  n | any_brake | hard  | med   | light | coast
---------- | -- | --------- | ----- | ----- | ----- | -----
 8-12 m/s  |479 |    92.7%  | 35.7% | 42.6% | 14.4% |  7.1%
12-18 m/s  |439 |    80.6%  | 22.1% | 28.5% | 30.1% | 18.7%
18-30 m/s  | 24 |    83.3%  |  4.2% |  8.3% | 70.8% | 16.7%
```

Early pedal-based analysis claimed "14% braking at high speed" — **wrong**,
it was a pedal-switch artifact. Actual aEgo shows 83% braking even at
18-30 m/s. The difference is **intensity**: almost entirely light braking
(71%) at speed vs hard (36%) at low speed.

This supports a speed-dependent >=0.80 gate: gentle capped brake at highway
speeds with moderate TTC/THW, full hazard handoff at low speeds.

---

## Pedal vs aEgo: the misclassification

In the 0.35–0.80 reqDecel band (n=573):

| Pedal label | n | aEgo says |
|---|---|---|
| "coast" | 268 | 63% actually braking (light), 34% truly coasting |
| "accel" | 85 | 82% truly coasting, 7% braking, 11% accelerating |

The pedal-based conclusion that humans "mostly coast" through 0.35–0.80 was
a labeling artifact. The controller commands acceleration, not pedal state.
All subsequent analysis uses aEgo as the primary truth.

---

## Derived tuning constants

### Coast / lift-off zone

Coast exceeds brake at TTC ≈ 16–18 s. The far-comfort gate at TTC > 24 s
is well into coast territory (68%). `COAST_TTC_MAIN = 20` captures the
upper end of the coast-preference region. At very high TTC (>24) with
comfortable headway, the system ignores (lets progress candidate dominate).

### Gentle-brake zone

From 0.10 to 0.50 required decel, human braking ramps from 34% to 62%.
The linear interpolation from 0.0 to -0.35 m/s² maps this range smoothly.
`COMFORT_REQUIRED_DECEL = 0.50` sets the ramp endpoint at the point where
humans are in clear majority-brake territory.

### Capped advisory zone (0.50–0.80)

Humans brake 75% here, mostly light-to-medium. A capped gentle brake at
-0.35 m/s² fills the former silent gap without overshooting.

### Hazard zone (>=0.80)

Humans brake 82–96% here, with intensity increasing as medTTC drops from
6s to 3.7s. Safety guard: TTC < 4 → always IGNORE (MPC owns it). At
highway speeds (vEgo >= 18, TTC >= 6, THW >= 1.2), humans use mostly
light braking → capped gentle brake. Otherwise → IGNORE.

### MIN_EXCESS_GAP

The 3–5 m excess gap band contains 413 closing events with 50% human
braking. Raising the margin would discard valid targets. Keep at 3 m.

### GENTLE_BRAKE_MAX

Human light-braking (aEgo -0.6 to -0.3) has a median of -0.409 m/s² and
75th percentile of -0.355 m/s². `-0.35` lies at the 75th percentile
(gentle end) — appropriate for a comfort advisory.

### Final constants

```python
_ALIGN_MIN_EXCESS_GAP         = 3.0
_ALIGN_TINY_REQUIRED_DECEL    = 0.10
_ALIGN_COMFORT_REQUIRED_DECEL = 0.50
_ALIGN_MAX_REQUIRED_DECEL     = 0.80
_ALIGN_GENTLE_BRAKE_MAX       = -0.35
_ALIGN_COAST_A_TARGET         = 0.0

_ALIGN_NO_ADVISORY_TTC        = 24.0
_ALIGN_COAST_TTC_MAIN         = 20.0
_ALIGN_STRONG_PREP_TTC        = 6.0
_ALIGN_HAZARD_TTC             = 4.0
```

---

## Caveats

- **Single driver.** All data is from one driver's daily commute. Braking
  style, following distance, and aggressiveness may not generalize.
- **qlog decimation (~7 Hz).** Radar samples are decimated; instantaneous
  dRel/vRel snapshots may miss lead-track transitions.
- **Binary brake switches.** `brakePressed` is a switch, not a pressure
  sensor — light trail-braking and panic stops are indistinguishable.
- **vEgo >= 8 m/s.** Excludes stop-and-go, merges, low-speed cut-ins, and
  any scenario where TTC is meaningless at low speed.
- **Lead track uncertainty.** No persistence across samples — a single high-
  vRel spike could be a lead changing lanes, not a hard braking event.
- **Secondary hazards.** Only `leadOne` is analyzed. A secondary threat
  (leadTwo, pedestrian, model-brake) is invisible to this analysis.

---

## Related files

- `sunnypilot/custom/longitudinal/lead_speed_alignment.py` — the tuned helper
- `sunnypilot/custom/longitudinal/tests/test_lead_speed_alignment.py` — unit tests
- `docs/plans/2026-06-12-fork-restart-reimplementation.md` — restart plan
- `/tmp/opencode/audit_analysis_aego.py` — the full analysis script
- `/tmp/opencode/sunnypilot-route-logs/` — raw qlog data
