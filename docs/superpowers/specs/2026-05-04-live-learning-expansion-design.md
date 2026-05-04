# Live Learning Parameter Expansion — Design Spec

## 1. Background & Motivation

sunnypilot currently has three live-learning subsystems:
- **torqued** — learns `latAccelFactor`, `latAccelOffset`, `frictionCoefficient` for lateral torque control
- **paramsd** — learns `steerRatio`, `stiffnessFactor`, `angleOffset`, `roll`
- **lagd** — learns steering actuation delay (`lateralDelay`)

These work well for global/static parameters but do not adapt to:
- **Speed-dependent steering dynamics** (rack assist, tire slip angles)
- **Changing vehicle mass / aero drag** (cargo, passengers, roof boxes, headwinds)
- **Non-linear brake/gas response curves** (pedal mapping, temperature, wear)

This spec adds three new extension-based live learners to address these gaps.

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

## 4. Sub-Project 2: Longitudinal Mass & Aero Drag Learning

### 4.1 Problem
`CP.mass` is static. Real-world mass changes (passengers, cargo) and drag changes (roof boxes, headwinds) cause feedforward errors, leading to sluggish acceleration or overshoot.

### 4.2 Design

**Physics Model**:
```
a_ego ≈ (k_force * a_cmd) - c_drag * v_ego² - c_roll
```
- `k_force` — effective drivetrain gain (absorbs mass and gear ratio)
- `c_drag` — aero drag coefficient (CdA / effective_mass)
- `c_roll` — rolling resistance / mass (assumed known or learned separately)

**Data Collection** (`sunnypilot/selfdrive/controls/lib/longitudinal_planner_ext.py`):
- Collect `(v_ego, a_cmd, a_ego)` during clean windows:
  - No lead car (`leadOne.status == False`)
  - Flat road (`abs(roll) < 1°`)
  - Low lateral accel (`|a_y| < 0.5 m/s²`)
  - `abs(a_cmd) > 0.1` or steady coasting for `> 2s`

**Estimation**:
- **Recursive Least Squares (RLS)** with forgetting factor λ = 0.995
- State vector: `[k_force, c_drag]`
- Sanity bounds: `0.5 ≤ k_force ≤ 2.0`, `c_drag ≥ 0`
- Reset on NaN or out-of-bounds

**Consumption** (`sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`):
- Adjust feedforward:
  ```
  ff_adjusted = a_target / k_force_learned
  ```
- Optionally subtract estimated drag from setpoint for MPC.

### 4.3 Params / UI
- `LongLearnedMassDragToggle` (bool, persistent)
- `LongLearnedKForce` (float, persistent)
- `LongLearnedCDrag` (float, persistent)
- Developer UI overlay showing learned vs nominal mass/drag

---

## 5. Sub-Project 3: Brake/Gas Response Curve Learning

### 5.1 Problem
The `output_accel` → `a_ego` mapping is non-linear and car-specific. Current `feedforward = a_target` assumes a 1:1 mapping, causing under/overshoot at different accel magnitudes.

### 5.2 Design

**Data Collection** (`sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`):
- Bucket `a_cmd` vs `a_ego` (delayed by `CP.longitudinalActuatorDelay` + `DT_CTRL`)
- Bucket bounds: `[-4.0, -2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0, 4.0]` m/s²
- Only collect when:
  - `long_control_state == pid`
  - Not actuator-saturated
  - `abs(a_ego - a_target) < 1.0` (steady-ish)

**Estimation**:
- Per-bucket mean offset: `offset = mean(a_ego - a_cmd)`
- First-order filter smoothing (decay = 100)

**Consumption**:
- Apply inverse mapping to feedforward:
  ```
  ff = a_target + lookup_offset(a_target)
  ```
- Clamp total correction to `±0.5 m/s²` per bucket to prevent instability.

### 5.3 Params / UI
- `LongLearnedResponseCurveToggle` (bool, persistent)
- `LongLearnedResponseOffsets` (JSON blob or comma-separated floats)

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
        ├──► longitudinal_planner.py + _ext ──► learned mass/drag
        │                                            │
        │                                            ▼
        └──► longcontrol.py + _ext ◄────────── response curve
                              │
                              ▼
                         carControl.actuators
```

---

## 8. Implementation Order

1. **Speed-Dependent Torque** — smallest change; reuses existing torqued infrastructure
2. **Response Curve** — isolated to longcontrol; easiest to validate
3. **Mass & Drag** — most complex; depends on clean data window logic

---

## 9. Testing Plan

- **Unit tests:** Mock `carState`/`carOutput`, verify RLS/bucket regression converges
- **Process replay:** Add learned params to replay baselines
- **Shadow logging:** Collect learned vs actual for 1000+ km before enabling override

---

## 10. Files to Modify / Create

| Path | Action |
|------|--------|
| `sunnypilot/selfdrive/locationd/torqued_ext.py` | Extend for speed buckets |
| `sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py` | Consume speed-aware params |
| `sunnypilot/selfdrive/controls/lib/longitudinal_planner_ext.py` | Add mass/drag RLS |
| `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py` | Add response curve + consume mass/drag |
| `common/params_keys.h` | Add new param keys |
| `sunnypilot/sunnylink/params_metadata.json` | Add descriptions |
| `selfdrive/ui/sunnypilot/.../torque_settings.py` | Add toggle UI |
| `selfdrive/ui/sunnypilot/.../device.py` or new settings page | Add longitudinal learner toggles |

---

## 11. Spec Self-Review Checklist

- [x] **Placeholder scan:** No TBDs, TODOs, or incomplete sections
- [x] **Internal consistency:** Architecture matches feature descriptions
- [x] **Scope check:** Three independent sub-projects clearly bounded
- [x] **Ambiguity check:** All requirements are explicit (bucket bounds, sanity ranges, thresholds)

---

*Spec written: 2026-05-04*
