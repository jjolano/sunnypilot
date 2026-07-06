# Episode detector recipes

All operate on the npz from `scripts/extract_route_npz.py`. Common prelude:

```python
import json, numpy as np
z = np.load("route.npz"); ev = json.load(open("route_events.json"))
t0 = z["cs_t"][0]
cs_t = z["cs_t"] - t0  # same for cc_/rs_/sp_/lp_/mv_
def interp(tq, ts, xs): return np.interp(tq, ts, xs)
def interp_bool(tq, ts, xs): return np.interp(tq, ts, xs.astype(float)) > 0.5
intent_names = {c: n for n, c in ev["intent_codes"].items()}
```

Channels: `cs_` carState 100Hz, `cc_` carControl 100Hz, `rs_` radarState leadOne 20Hz,
`lp_` longitudinalPlan 20Hz, `sp_` longitudinalPlanSP (+`dbg*` = longitudinalDebug) 20Hz,
`mv_` modelV2 lanes 20Hz, `ss_` selfdriveState. Align with `interp*` onto the query
timebase; group hit indices into episodes with `np.split` on `np.diff(idx) > gap`.

## Stopped-behind-lead gap (engaged vs manual)

Mask: `standstill & leadStatus & vLead<0.3`, split by `longActive`. Quote median/p10/p90
dRel for both. Per-episode: group and print `dRel start/min/end` — a growing dRel during
an engaged episode is the no-crawl evidence.

## Crawl-up check

Within engaged-standstill-with-stopped-lead episodes, find gap growth ≥2 m from episode
start; check whether ego moved (v>0.2) within 4 s of the gap opening.

## Brake/gas interventions

Onsets: `np.diff(brake.astype(int))==1` while `enabled` (interp to cs timebase). For each
hit print the context line (finalA, clip, modelA, mpcA, custA, intent, vTarget) sampled
0.2–0.3 s *before* the press — the post-press frames already reflect the override.
A pinned custA at a comfort constant while modelA ramps negative = policy under-executing
the model stop. Intent flapping cruise↔stop_approach frame-to-frame = missing hysteresis.

## Lead cut-out overbraking

Hits: leadOne `status` drop or dRel jump > 8 m between consecutive radar frames while
`longActive & v>3 & dRel<45`. Evidence of a real cut-out: yRel trending beyond ±1.5–3 m
in the 2.5 s before the drop. Overbrake if min `cc_accel` in [-3s,+1.5s] < -0.8 **and**
`sp_vTarget` is not pinned at an SCC floor (else it's an SCC slowdown, different owner).

## Launches (engaged vs manual)

Standstill→moving edges (`np.diff(standstill)== -1`). Per launch: engaged iff longActive
and no gas in [-1s,+2s]; lead-reaction delay = launch time minus first `vLead>0.5` in the
prior 15 s; report meanA(0–2s), peakA(0–4s), v@4s. For engaged launches also dump the sp
timeline: hold-release lag (finalA leaves -2.0), then which cap binds (a plateau at
`LEAD_CRAWL_ACCEL_MAX` shows as finalA≈0.36–0.55 while mpcA is higher).

## Wander (straight-road oscillation)

`center_off = (llInnerLeftY + llInnerRightY)/2` where both probs > 0.5, latActive,
v > 8, no blinker, **and no laneChange event within the window**. 20–30 s windows,
straightness gate `|desCurv|.mean() < 0.004`. Report p2p (p98−p2) and the 0.05–0.3 Hz
band share via FFT. Known signature: 0.05–0.15 Hz, p2p 0.6–0.9 m = model limit cycle.

## Lane-line proximity

Clearance = `-llInnerLeftY - HALF_W` / `llInnerRightY - HALF_W` (HALF_W ≈ 0.95 m RAV4).
Same gating as wander. Quote median/p5 per side, % time < 0.25 m, and episodes < 0.15 m
(≥4 consecutive frames). Verify worst episodes against laneChange events before calling
them drift; a snap of both lines plus prob collapse is a lane change, not an excursion.

## Full-context frame dump

For any timestamp of interest print every ~0.2–0.5 s in a ±5 s window:

```
t v aEgo gas brk en longAct | dRel yRel vLead | finalA clip[min,max] modelA mpcA custA mStop intent vTarget
```

This single line format resolved every complaint in the route-261 diagnosis; see
memory `route-261-longitudinal-lateral-diagnosis` for worked examples.
