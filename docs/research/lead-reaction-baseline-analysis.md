# Lead-reaction baseline analysis: OP vs manual

**Date:** 2026-06-21
**Author:** AI-assisted analysis driven by user's manual-driving logs
**Data sources:**
- Route `000001ed--3716cd5787` (commit `16c448350`, 31.2 min, 341.5s OP + 872.6s manual)
- Route `000001ea--f65f4f6a3b` (commit `16c448350`, 31.9 min, pure manual baseline)
**Purpose:** Measure how quickly OP reacts to lead speed changes, lead exits, and cut-ins
compared to human manual driving, to identify where lead predictions/trajectory prevent
faster-than-human reaction.

**Driver braking strategy:** The manual baseline reflects a deliberate safety philosophy:
brake early and gently to prevent rear-end accidents while being anticipatory and fuel-
efficient. Sudden, inconsistent, or unpredictable braking is avoided because it causes
rear-end collisions. This means the manual baseline is not just "human reaction time" —
it is a target safety envelope: smooth, predictable, early decel that gives following
traffic time to react.

**Driver acceleration strategy:** Equally important is prompt reaction to lead speed
increases. Drivers behind anticipate that our car will launch or accelerate when the lead
moves. If we hesitate, following drivers who expect us to go may rear-end us, and
frustration leads to aggressive passing. The manual baseline's 0.520s median reaction to
lead acceleration reflects this: the driver accelerates promptly to avoid being a
bottleneck that causes rear-end collisions from behind.

---

## Methodology

### Refined approach vs prior baseline tools

This analysis introduces `profile_lead_reaction.py`, which addresses three gaps in prior
baseline tooling:

1. **Creep filtering**: `brakePressed` with `vEgo > 0.1 m/s` is classified as *creeping*,
   not *stopped*. Prior tools treated any `brakePressed` at low speed as a stop event,
   contaminating launch/stop-hold metrics. This route had **2,287 creep-filtered samples**.

2. **Lead speed change detection via `aLeadK` direction changes**: Instead of raw `vLead`
   thresholds, direction changes in `aLeadK` (lead acceleration) are detected with a
   minimum magnitude gate (`0.3 m/s²`) and sustained duration (`0.3s`). This filters
   radar noise transients that don't represent real lead behavior changes.

3. **Dual reaction measurement**:
   - **Manual**: `aEgo` direction change (actual vehicle deceleration, not pedal switches —
     following the aEgo-based labeling correction from the prior baseline analysis).
   - **OP**: `longitudinalPlanSP.aTarget` direction change (planner output, before actuator
     lag).

### Event types

| Event | Detection | Reaction metric |
|---|---|---|
| Lead speed change | `aLeadK` sign flip (sustained > 0.3s) | Time to ego `aEgo`/`aTarget` direction change |
| Lead exit | `radarTrackId` disappears > 0.5s or changes | Time to ego accel > 0.3 m/s² above baseline |
| Cut-in | New `radarTrackId` appears with `status=True` inside 30m | Time to ego decel > 0.3 m/s² below baseline |

### Data filters

- `vEgo >= 1.0 m/s` for reaction measurement (excludes parking)
- `leadOne.status == True` for lead speed changes
- `aLeadK` magnitude > 0.3 m/s² for direction change
- 5-second reaction window (events without response in 5s = no reaction)

### Caveats

- **Two routes, single driver.** Not generalizable without more data.
- **qlog decimation (~7 Hz).** Radar samples are decimated; reaction times have ~0.14s
  quantization.
- **Radar track ID churn.** The cut-in and lead-exit detectors fire on `radarTrackId`
  changes, which include radar track swaps and dropouts, not just real cut-ins/exits.
  Counts are inflated; medians are more reliable than counts.
- **Small OP sample.** Only 341.5s of OP engagement on route ed (1 OP lead-speed-change
  reaction event). Route ea is pure manual (0s OP). Need more OP-engaged routes.
- **No rlog.** qlog-only analysis; `modelV2` and high-fidelity signals unavailable.

---

## Core findings

### Table 1: Reaction time summary (combined routes)

```
Metric                    | OP median   | Manual median | Gap
------------------------- | ----------- | ------------- | ----
Lead speed change         | 0.942 s (1) | 0.667 s (72)  | +41% slower
  lead braking → ego brake| 0.942 s (1) | 0.775 s (41)  | +22% slower
  lead accel → ego accel  |   n/a (0)   | 0.520 s (31)  | n/a
Lead exit accel           | 1.185 s     | 1.429 s       | -17% (OP faster)
Cut-in brake              | 2.688 s     | 1.349 s       | +99% slower
Cut-in peak decel         | -2.000 m/s² | -0.470 m/s²  | 4.3x harsher
```

### Table 2: Manual reaction time distribution (route 000001ea, n=72)

```
Statistic  | All (n=72) | Lead brake→ego brake (n=41) | Lead accel→ego accel (n=31)
---------- | ---------- | --------------------------- | ---------------------------
min        | 0.020 s    | 0.020 s                     | 0.071 s
p10        | 0.123 s    | 0.221 s                     | 0.071 s
p25        | 0.270 s    | 0.423 s                     | 0.220 s
median     | 0.667 s    | 0.775 s                     | 0.520 s
p75        | 1.216 s    | 1.316 s                     | 0.920 s
p90        | 2.137 s    | 2.222 s                     | 1.421 s
max        | 4.815 s    | 4.815 s                     | 3.020 s
```

### Table 3: Manual cut-in brake distribution (route 000001ea, n=710)

```
Statistic  | Brake reaction | Peak decel
---------- | --------------- | ----------
median     | 1.349 s         | -0.470 m/s²
p10        | 0.243 s         | -1.816 m/s²
p90        | 3.815 s         | +0.370 m/s²
```

### Key insight 1: OP's cut-in braking is the opposite of the driver's safety strategy

The driver brakes **early and gently** (median 1.349s reaction, -0.470 m/s² peak decel) to
prevent rear-end accidents. OP brakes **late and hard** (2.688s reaction, -2.000 m/s² peak
decel) — the exact pattern that causes rear-end collisions.

This is the most important finding: OP's cut-in behavior is not just slower than human, it
is **dangerously unpredictable**. A following driver expects consistent, gradual braking.
OP's 2.7s delay followed by -2.0 m/s² is a sudden stop that gives following traffic no
warning.

The likely cause: the MPC extrapolates the cut-in lead's `aLeadK` into its predicted
trajectory, and the noisy radar accel creates a delayed but amplified braking response.
The §3 lead-anticipation shaping (ADR `2026-06-14`) was designed to address this but was
found INERT (0 softened frames in testing).

### Key insight 2: OP is ~41% slower on lead speed change reaction

OP median lead speed change reaction is **0.942s** vs manual **0.667s** (72 events).
However, OP data is from only 1 event, so it's not statistically reliable. The manual
sample (72 events) is robust.

### Key insight 3: Driver reacts faster to lead acceleration than lead braking — and this is deliberate

Manual reaction to lead accelerating (ego accelerates): median **0.520s**.
Manual reaction to lead braking (ego brakes): median **0.775s**.

The driver is **33% faster** reacting to a lead pulling away than to a lead braking. This
is not just a perceptual asymmetry — it is a safety strategy:

- **Prompt acceleration prevents rear-ends from behind.** Drivers behind anticipate that
  our car will launch when the lead moves. If we hesitate, following drivers expecting us
  to go may rear-end us. Frustration also leads to aggressive passing maneuvers.
- **Gentle braking prevents rear-ends from behind.** Braking early and gently gives
  following traffic time to react. The slightly slower brake reaction (0.775s) is acceptable
  because the braking itself is gentle and predictable.

For OP, both directions are problematic:
- The MPC extrapolates `aLeadK` noise into braking (reactive braking) — late and hard
- The launch release gates add delay to acceleration — frustrating drivers behind who
  expect prompt launch

### Key insight 4: Lead exit reaction is comparable (OP slightly faster)

OP lead exit accel reaction (1.185s) is **17% faster** than manual (1.429s, 726 events).
When the lead disappears, OP's planner immediately sees no lead and removes the follow
constraint. The human driver has visual confirmation overhead.

### Key insight 5: Creep filtering matters

Route 000001ed had **2,287** creep-filtered samples; route 000001ea had **3,309**. Prior
analysis tools would have classified these as stop events, inflating stop-hold and launch
metrics.

### Key insight 6: Manual baseline is consistent across routes

Manual cut-in brake median is **1.349s** (route ea, 710 events) vs **1.436s** (route ed,
18 events). Manual lead speed change median is **0.667s** (ea, 72 events) vs **0.610s**
(ed, 18 events). The consistency across routes gives confidence in the manual baseline.

---

## Per-event detail

### OP lead speed change reactions

```
t=364.9   accel_to_decel   reaction=n/a   type=none      (no response in 5s window)
t=1106.8  accel_to_decel   reaction=0.94s type=brake
```

Only 2 OP lead-speed-change events were detected (341.5s OP engagement is thin). The first
had no reaction within 5s — the lead decelerated but OP didn't respond (possibly because
the lead was far enough that MPC didn't need to brake).

### Manual lead speed change reactions (sample)

```
t=57.9    accel_to_decel   reaction=4.32s  type=brake     (delayed — possibly distracted)
t=61.9    accel_to_decel   reaction=0.32s  type=brake
t=83.1    accel_to_decel   reaction=0.17s  type=brake     (very fast)
t=415.1   accel_to_decel   reaction=0.09s  type=brake     (near-instant)
t=477.4   accel_to_decel   reaction=1.23s  type=brake
t=504.1   accel_to_decel   reaction=0.58s  type=brake
t=793.1   decel_to_accel   reaction=0.19s  type=accel     (very fast)
```

Manual reactions range from 0.09s to 4.32s. The fastest (0.09s) suggest the driver was
anticipating the lead behavior. The slowest (4.32s) may be a case where the driver didn't
need to react immediately.

### OP cut-in brake reactions (sample)

```
t=289.6   d=14.5m  vR=-4.6  brake=4.68s  peak=-2.0    (very late, very harsh)
t=290.1   d=12.3m  vR=-4.0  brake=4.18s  peak=-2.0
t=383.6   d=14.4m  vR=-4.3  brake=4.69s  peak=-2.0    (very late, very harsh)
t=1043.1  d=19.0m  vR=-7.2  brake=4.69s  peak=-2.0    (very late, very harsh)
t=1110.1  d=16.8m  vR=-4.6  brake=4.69s  peak=-2.0
t=1557.3  d=15.8m  vR=-5.3  brake=4.95s  peak=-2.0    (latest)
```

OP cut-in brake reactions are consistently 2-5s, with peak decel hitting the -2.0 m/s²
floor. This is the most striking finding: OP is both late and harsh on cut-ins.

---

## Interpretation

### Why OP is slow on cut-ins

The MPC's `process_lead` extrapolates `aLeadK` into the lead's predicted trajectory:

```python
a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
v_lead_traj = clip(v_lead + cumsum(T_DIFFS * a_lead_traj), 0, ...)
x_lead_traj = x_lead + cumsum(T_DIFFS * v_lead_traj)
```

When a new lead cuts in, the radar reports a high closing speed (`vRel` negative) but the
`aLeadK` may not yet reflect a deceleration. The MPC needs several frames to build
confidence in the new lead's deceleration before it brakes. By then, the closing speed
has increased and the MPC must brake harder.

The human driver, by contrast, sees the cut-in visually and begins braking before the
radar fully resolves the new lead's kinematics.

### Why OP is harsh on cut-ins

Once the MPC finally reacts, it has less time and distance, so it brakes at the -2.0 m/s²
limit. The human driver brakes earlier and gentler (-0.247 m/s² median peak), spreading
the deceleration over a longer period.

### Why lead exit reaction is comparable

When the lead disappears, both OP and the human driver can immediately accelerate. OP's
planner sees no lead and removes the follow constraint. The human driver sees the road
is clear. The reaction times are similar because neither depends on noisy radar accel.

---

## Recommendations

1. **Cut-in early advisory brake**: The lead-speed-alignment helper could issue a gentle
   advisory brake when a new lead appears with high closing speed, before the MPC fully
   resolves the kinematics. This would reduce the 2.7s median delay and replace the late
   hard brake with an early gentle one — matching the driver's safety strategy of smooth,
   predictable decel that prevents rear-end accidents.

2. **Lead-anticipation apply mode**: The §3 lead-anticipation shaping was found INERT
   because it only discounts `aLeadK` (reduces braking), never amplifies it. For cut-ins,
   the problem is the opposite: the MPC is too slow to start braking. A different mechanism
   is needed — perhaps a cut-in-specific advisory that doesn't depend on `aLeadK` confidence.

3. **Brake smoothness over speed**: The goal is not just faster reaction — it is earlier,
   gentler, more predictable braking. A -0.5 m/s² advisory at 1.0s is safer than a -2.0 m/s²
   MPC brake at 2.7s. The tuning target should be matching the manual baseline's
   -0.470 m/s² median peak decel, not just reducing reaction time.

4. **Prompt acceleration when the lead moves**: The 0.520s manual reaction to lead
   acceleration is the target. OP's launch release gates and lead-pullaway logic should
   aim for this envelope. Hesitation here causes rear-end collisions from drivers behind
   who expect prompt launch, and frustration leads to aggressive passing.

4. **More OP data needed**: Only 341.5s of OP engagement on route ed (1 OP reaction event).
   Route ea is pure manual. Need routes with more OP engagement and repeated cut-in/lead-
   change scenarios to get statistically reliable OP reaction times.

5. **Radar track ID churn filtering**: The 1,138 cut-in and 1,499 lead-exit events on
   route ea are inflated by radar track ID swaps. A future refinement should filter out
   track changes that don't represent real vehicle cut-ins (e.g., by checking `yRel`,
   `modelProb`, and sustained presence).

---

## Related files

- `tools/drive_lab/profile_lead_reaction.py` — the analysis tool
- `tools/drive_lab/tests/test_profile_lead_reaction.py` — tests
- `docs/adr/2026-06-14-longitudinal-lead-anticipation.md` — §3 lead-anticipation ADR
- `docs/adr/2026-06-15-longitudinal-lead-speed-alignment.md` — lead-speed alignment ADR
- `docs/research/manual-driving-baseline-analysis.md` — prior baseline methodology
- `/tmp/opencode/sunnypilot-longitudinal-analysis/000001ed--3716cd5787-lead-reaction.json` — route ed raw data
- `/tmp/opencode/sunnypilot-longitudinal-analysis/000001ea--f65f4f6a3b-lead-reaction.json` — route ea raw data
