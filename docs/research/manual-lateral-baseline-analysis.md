# Manual lateral baseline analysis

**Date:** 2026-06-20
**Author:** AI-assisted analysis driven by user's device logs
**Data source:** Personal comma device, qlogs from recent daily driving
**Routes:** 8 routes, 216 segments, 79,012 accepted lateral samples
**Purpose:** Build an evidence-only lateral comparison analogous to the recent
manual longitudinal baseline work, without treating manual steering as a control
target or deriving live tuning changes.

---

## Methodology

### Data collection

Recent qlogs were read from `/data/media/0/realdata` on the device. Local copies
were analyzed from `/tmp/opencode/sunnypilot-route-logs`. The two newest routes
(`000001e9--3c6c94441c`, `000001ea--f65f4f6a3b`) were pulled for this pass; the
remaining six were already present from the longitudinal analysis corpus.

### Sample filters

The analysis uses `controlsState` samples with the latest `carState`,
`carControl`, and `modelV2` context.

| Filter | Rationale |
|---|---|
| `vEgo >= 3.0 m/s` | Exclude near-standstill steering noise |
| no blinkers | Avoid lane changes / merge intent |
| no lane-change state when known | Avoid comparing lane-change behavior to lane-keep behavior |
| no standstill | Avoid parked / stopping steering artifacts |
| engaged requires `carControl.latActive` and lateral state active | Strict split between system and manual |
| engaged excludes `steeringPressed` | Driver override is not controller evidence |

Manual lateral samples are descriptive only. They form a style/context envelope,
not a ground truth lane-center target.

### Metrics

Each accepted sample computes:

- current curvature and actual lateral acceleration
- desired lateral acceleration from logged torque-state telemetry when engaged
- engaged lateral acceleration error = desired − actual
- approximate lateral jerk
- approximate steering rate
- speed bins: 3–8, 8–12, 12–18, 18–30, >=30 m/s
- lateral-accel magnitude bins: 0–0.3, 0.3–0.8, 0.8–1.5, >=1.5 m/s²
- curve side: left / right / straight

For engaged routes, existing Drive Lab lateral gates were also run:

- `lateral_performance_gate`
- `lateral_torque_event_report`
- `lateral_low_speed_report`

---

## Corpus split

```
routes:        8
segments:      216
samples:       79,012
manual:        65,704
engaged:       13,308
```

| Route | Segments | Manual samples | Engaged samples | Notes |
|---|---:|---:|---:|---|
| `000001e3--e1e1edacd2` | 54 | 25,202 | 0 | manual-only |
| `000001e4--053b71ba66` | 27 | 10,395 | 0 | manual-only |
| `000001e5--27ff1f95cd` | 31 | 10,321 | 0 | manual-only |
| `000001e6--32aeed6695` | 12 | 66 | 2,996 | engaged-heavy |
| `000001e7--0b73fed711` | 22 | 581 | 5,673 | engaged-heavy |
| `000001e8--7b3f4ae697` | 12 | 9 | 4,639 | engaged-heavy |
| `000001e9--3c6c94441c` | 26 | 9,041 | 0 | manual-only |
| `000001ea--f65f4f6a3b` | 32 | 10,089 | 0 | manual-only |

The split is uneven: manual and engaged evidence mostly comes from different
routes. This means the corpus supports envelope/risk observations, not a direct
route-matched “manual did X, system should do X” conclusion.

---

## Manual lateral envelope

Route-level p95 absolute actual lateral acceleration, m/s²:

| Route | 8–12 m/s | 12–18 m/s | 18–30 m/s |
|---|---:|---:|---:|
| `000001e3--e1e1edacd2` | 0.146 | 0.813 | 1.009 |
| `000001e4--053b71ba66` | 0.169 | 0.245 | 0.127 |
| `000001e5--27ff1f95cd` | 0.518 | 0.582 | 0.189 |
| `000001e7--0b73fed711` | 1.099 | 0.182 | 0.108 |
| `000001e9--3c6c94441c` | 0.548 | 0.338 | 0.312 |
| `000001ea--f65f4f6a3b` | 0.700 | 0.725 | 0.277 |

Observed manual envelope is broad and route-dependent. It should be used to find
candidate speed/curve regimes for review, not as a steering-gain target.

---

## Engaged lateral tracking

Route-level p95 absolute engaged lateral acceleration error, m/s²:

| Route | 8–12 m/s | 12–18 m/s | 18–30 m/s |
|---|---:|---:|---:|
| `000001e6--32aeed6695` | 0.149 | 0.147 | 0.127 |
| `000001e7--0b73fed711` | 0.164 | 0.130 | 0.177 |
| `000001e8--7b3f4ae697` | 0.188 | 0.123 | n/a |

Route-level p95 absolute engaged actual lateral acceleration, m/s²:

| Route | 8–12 m/s | 12–18 m/s | 18–30 m/s |
|---|---:|---:|---:|
| `000001e6--32aeed6695` | 0.546 | 0.512 | 0.352 |
| `000001e7--0b73fed711` | 0.584 | 0.442 | 0.558 |
| `000001e8--7b3f4ae697` | 0.680 | 0.375 | n/a |

Initial read: engaged tracking error is not globally large in these route-level
bins. The more interesting evidence is event-local: the lateral gates identify
fast reversal and path/wander windows despite modest aggregate tracking error.

---

## Engaged route gate findings

| Route | Active % | Dominant class | Confidence | Torque score | Wander score | Low-speed score |
|---|---:|---|---|---:|---:|---:|
| `000001e6--32aeed6695` | 66.2 | path-wander dominant | medium | 62.8 | 75.4 | 40.4 |
| `000001e7--0b73fed711` | 66.4 | torque-event dominant | high | 72.4 | 30.6 | 46.8 |
| `000001e8--7b3f4ae697` | 85.0 | path-wander dominant | medium | 96.9 | 136.5 | 57.0 |

Representative event-local findings:

- `000001e6`: top wander window at 73.9–102.8s, medium-confidence
  actuation-driven straight-path wander, steering peak-to-peak ≈30.0°.
- `000001e7`: top torque event at 954.6–957.3s, demand-driven, steering
  peak-to-peak ≈20.0°, steering-rate p95 ≈69.8°/s.
- `000001e8`: top wander window at 156.4–186.3s, medium-confidence
  actuation-driven straight-path wander, steering peak-to-peak ≈53.7°.

Qlog lane-change state was unknown on these routes, so wander classifications use
qlog-safe handling and remain medium-confidence unless reviewed with richer rlogs
or video/context.

---

## Conclusions

1. The lateral corpus is usable for evidence triage, but not yet for direct
   manual-vs-engaged tuning because manual and engaged samples are mostly on
   different routes.
2. Aggregate engaged lateral acceleration error is modest in the 8–30 m/s bins
   sampled here: p95 is roughly 0.12–0.19 m/s² on engaged routes.
3. The main lateral signals are event-local: fast torque reversals and
   straight/path-wander windows, especially on `000001e7` and `000001e8`.
4. Manual lateral data should remain a comfort/style envelope. It does not
   justify live gain adaptation, torque boost, or governor relaxation.
5. Any under-response fix still needs engaged-only evidence that actual lateral
   acceleration trails processed demand under clean path-state conditions.

---

## Artifacts

- Analyzer: `tools/drive_lab/manual_lateral_baseline.py`
- Tests: `tools/drive_lab/tests/test_manual_lateral_baseline.py`
- Local JSON/text outputs: `/tmp/opencode/sunnypilot-lateral-analysis/`
- Route qlog corpus: `/tmp/opencode/sunnypilot-route-logs/`

Primary commands:

```bash
uv run --extra testing --extra tools python -m openpilot.tools.drive_lab.manual_lateral_baseline ROUTE --qlog --log-root /tmp/opencode/sunnypilot-route-logs
uv run --extra testing --extra tools python -m pytest tools/drive_lab/tests/test_manual_lateral_baseline.py -q
```
