# Live Learning Parameter Expansion — Design Spec

## 1. Background & Motivation

sunnypilot currently has three live-learning subsystems:
- **torqued** — learns `latAccelFactor`, `latAccelOffset`, `frictionCoefficient` for lateral torque control
- **paramsd** — learns `steerRatio`, `stiffnessFactor`, `angleOffset`, `roll`
- **lagd** — learns steering actuation delay (`lateralDelay`)

These work well for global/static parameters but do not adapt to:
- **Speed-dependent steering dynamics** (rack assist, tire slip angles)
- **Changing vehicle mass / aero drag** (cargo, passengers, roof boxes, headwinds)

This spec adds two active extension-based live learners to address these gaps.

---

## 2. Scope & Architecture

All three features follow **Approach A (Extension-Based)**:
- Extend existing processes with sunnypilot extension files (`.py` in `sunnypilot/.../`)
- No new cereal services for MVP
- New params in `common/params_keys.h` and `params_metadata.json`
- Gated behind user toggles

---

## 3. Sub-Project 1: Speed-Dependent Torque Learning

### 3.1 Problem
A single global `latAccelFactor` is inaccurate across the speed range. EPS assist curves and tire dynamics vary significantly between 5 m/s and 40 m/s.

### 3.2 Design

**Data Collection** (`sunnypilot/selfdrive/locationd/torqued_ext.py`):
- Add `v_ego` tag to each torque-steering point in `TorqueBuckets`.
- Segment points into speed buckets: `[0, 10, 20, 30, 40+]` m/s.
- Maintain a separate `TorqueBuckets` instance per speed band (or filter by speed during estimation).

**Estimation**:
- When any speed bucket has `> MIN_POINTS_TOTAL_QLOG` points, run the existing TLS regression to get `latAccelFactor_speed`.
- Apply per-bucket sanity clamps and first-order decay filters.
- Global factor remains as fallback for uncalibrated buckets.

**Consumption** (`sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py`):
- In `update_override_torque_params`, interpolate `latAccelFactor` across speed buckets using `CS.vEgo`.
- Use global factor if interpolated bucket lacks calibration.

### 3.3 Params / UI
- `LiveTorqueSpeedAdaptiveToggle` (bool, persistent)
- Display per-bucket calibration percentage in developer UI

---

## 4. Removed: Longitudinal Mass & Aero Drag Learning

Mass/drag learning was removed from the active design. The RLS estimator, long-control feedforward compensation, UI toggles, params, metadata, and tests are no longer retained because the learned drag term was not expected to be stable enough for control use.

---

## 5. Removed: Brake/Gas Response Curve Learning

Brake/gas response-curve learning was removed from the active design. The control path now keeps `feedforward = a_target`, and no response-curve params, UI toggles, or learned offset cache are retained.

---

## 6. Safety & Error Handling

| Mechanism | Description |
|-----------|-------------|
| Sanity Clamping | Every learned param clamped to safe range before use |
| Invalidation | If variance > threshold or NaN, reset to default & clear cache |
| Calibration Threshold | Learned values only active after `calPerc > 50%` |
| Toggle Gating | Each learner independently enable/disable; off-road setting changes |
| Shadow Mode | First release: learners log but do not override control |

---

## 7. Data Flow

```
carState, carOutput, livePose, liveDelay
        │
        ├──► torqued.py + torqued_ext.py ──► liveTorqueParameters
        │                                        │
        │                                        ▼
        │                              latcontrol_torque.py + _ext
        │
        └──► longitudinal_planner.py
                                                     │
                                                     ▼
                                                longcontrol.py
                              │
                              ▼
                         carControl.actuators
```

---

## 8. Implementation Order

1. **Speed-Dependent Torque** — retained; reuses existing torqued infrastructure
2. **Response Curve** — removed from the active design
3. **Mass & Drag** — removed from the active design

---

## 9. Testing Plan

- **Unit tests:** Verify speed-bucket torque learning and removed longitudinal learners stay absent
- **Process replay:** Validate retained learned params against replay baselines
- **Shadow logging:** Collect retained speed-aware torque learning before enabling apply paths

---

## 10. Files to Modify / Create

| Path | Action |
|------|--------|
| `sunnypilot/selfdrive/locationd/torqued_ext.py` | Extend for speed buckets |
| `sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py` | Consume speed-aware params |
| `common/params_keys.h` | Keep retained speed-aware torque params only |
| `sunnypilot/sunnylink/params_metadata.json` | Keep retained speed-aware torque metadata only |
| `selfdrive/ui/sunnypilot/.../torque_settings.py` | Add retained torque learning toggle UI |

---

## 11. Spec Self-Review Checklist

- [x] **Placeholder scan:** No TBDs, TODOs, or incomplete sections
- [x] **Internal consistency:** Architecture matches feature descriptions
- [x] **Scope check:** Three independent sub-projects clearly bounded
- [x] **Ambiguity check:** All requirements are explicit (bucket bounds, sanity ranges, thresholds)

---

*Spec written: 2026-05-04*
