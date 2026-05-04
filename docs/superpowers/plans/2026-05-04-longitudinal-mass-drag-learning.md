# Longitudinal Mass & Aero Drag Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn effective vehicle mass (as a drivetrain gain scalar) and aerodynamic drag coefficient online, improving longitudinal feedforward accuracy when vehicle loading or conditions change.

**Architecture:** Extend `longitudinal_planner.py` with a recursive least-squares (RLS) estimator that observes `(v_ego, a_cmd, a_ego)` during clean driving windows. Store learned `k_force` and `c_drag` in Params. `longcontrol.py` extension adjusts feedforward using learned values.

**Tech Stack:** Python 3, NumPy, openpilot Params, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `sunnypilot/selfdrive/controls/lib/longitudinal_planner_ext.py` | RLS estimator, clean window detection, param writes |
| `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py` | Consumes learned mass/drag, adjusts feedforward |
| `selfdrive/controls/lib/longitudinal_planner.py` | Wires extension into update loop |
| `selfdrive/controls/lib/longcontrol.py` | Wires extension into PID update |
| `selfdrive/controls/tests/test_long_learned_mass_drag.py` | Unit tests for RLS and window logic |
| `common/params_keys.h` | New param key definitions |
| `sunnypilot/sunnylink/params_metadata.json` | Param descriptions |

---

## Prerequisites

Ensure you have read:
- `docs/superpowers/specs/2026-05-04-live-learning-expansion-design.md` Section 4
- `selfdrive/controls/lib/longitudinal_planner.py` — understand `LongitudinalPlanner.update()`
- `selfdrive/controls/lib/longcontrol.py` — understand `LongControl.update()` and feedforward
- `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` — understand MPC output

---

## Task 1: Add Param Keys and Metadata

**Files:**
- Modify: `common/params_keys.h`
- Modify: `sunnypilot/sunnylink/params_metadata.json`

- [ ] **Step 1: Add param keys**

In `common/params_keys.h`, add:

```c
    {"LongLearnedMassDragToggle", {PERSISTENT | BACKUP, BOOL, "0"}},
    {"LongLearnedKForce", {PERSISTENT, FLOAT, "1.0"}},
    {"LongLearnedCDrag", {PERSISTENT, FLOAT, "0.0"}},
```

- [ ] **Step 2: Add metadata**

In `sunnypilot/sunnylink/params_metadata.json`:

```json
  "LongLearnedMassDragToggle": {
    "title": "Learn Mass & Drag",
    "description": "Live-learns vehicle mass and aerodynamic drag to improve acceleration and braking accuracy."
  },
  "LongLearnedKForce": {
    "title": "Learned Drivetrain Gain",
    "description": ""
  },
  "LongLearnedCDrag": {
    "title": "Learned Drag Coefficient",
    "description": ""
  },
```

- [ ] **Step 3: Commit**

```bash
git add common/params_keys.h sunnypilot/sunnylink/params_metadata.json
git commit -m "params: add LongLearnedMassDragToggle, KForce, and CDrag"
```

---

## Task 2: Implement RLS Dynamics Estimator

**Files:**
- Create: `sunnypilot/selfdrive/controls/lib/long_learned_mass_drag.py`
- Create: `selfdrive/controls/tests/test_long_learned_mass_drag.py`

- [ ] **Step 1: Write the failing test**

Create `selfdrive/controls/tests/test_long_learned_mass_drag.py`:

```python
#!/usr/bin/env python3
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.long_learned_mass_drag import RLSDynamicsEstimator


def test_rls_converges_to_known_params():
  estimator = RLSDynamicsEstimator(forgetting_factor=0.99)
  # Simulate data with k_force=0.8, c_drag=0.02
  np.random.seed(42)
  for _ in range(500):
    v = np.random.uniform(10, 30)
    a_cmd = np.random.uniform(-1, 2)
    a_ego = 0.8 * a_cmd - 0.02 * v**2 + np.random.normal(0, 0.05)
    estimator.update(v, a_cmd, a_ego)

  k_force, c_drag = estimator.get_params()
  assert 0.75 < k_force < 0.85
  assert 0.015 < c_drag < 0.025


def test_rls_ignores_invalid_data():
  estimator = RLSDynamicsEstimator()
  assert not estimator.is_valid()
  estimator.update(5.0, 0.5, 0.3)  # too slow
  assert not estimator.is_valid()


def test_rls_sanity_reset():
  estimator = RLSDynamicsEstimator()
  for _ in range(100):
    estimator.update(20.0, 1.0, 5.0)  # impossible a_ego -> should reset
  k_force, _ = estimator.get_params()
  assert k_force == 1.0  # default after reset
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/jjolano/Developer/projects/sunnypilot
pytest selfdrive/controls/tests/test_long_learned_mass_drag.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RLSDynamicsEstimator**

Create `sunnypilot/selfdrive/controls/lib/long_learned_mass_drag.py`:

```python
import numpy as np

MIN_VEGO = 10.0
MIN_POINTS_VALID = 50
KF_DEFAULT = 1.0
CDRAG_DEFAULT = 0.0
KF_MIN, KF_MAX = 0.5, 2.0
CDRAG_MIN, CDRAG_MAX = 0.0, 0.1


class RLSDynamicsEstimator:
  def __init__(self, forgetting_factor=0.995):
    self.lam = forgetting_factor
    # State: [k_force, c_drag]
    self.theta = np.array([[KF_DEFAULT], [CDRAG_DEFAULT]], dtype=np.float64)
    self.P = np.eye(2) * 10.0
    self.points = 0
    self._valid = False

  def update(self, v_ego, a_cmd, a_ego):
    if v_ego < MIN_VEGO:
      return

    # Regression model: a_ego = k_force * a_cmd - c_drag * v_ego^2
    phi = np.array([[a_cmd], [-v_ego ** 2]], dtype=np.float64)
    y = np.array([[a_ego]], dtype=np.float64)

    # RLS update
    P_phi = self.P @ phi
    denom = self.lam + phi.T @ P_phi
    K = P_phi / denom
    error = y - phi.T @ self.theta
    self.theta = self.theta + K @ error
    self.P = (self.P - K @ phi.T @ self.P) / self.lam

    # Sanity clamp and reset if needed
    k_force = float(self.theta[0, 0])
    c_drag = float(self.theta[1, 0])
    if not (KF_MIN <= k_force <= KF_MAX) or not (CDRAG_MIN <= c_drag <= CDRAG_MAX) or not np.isfinite(k_force) or not np.isfinite(c_drag):
      self.reset()
      return

    self.points += 1
    if self.points >= MIN_POINTS_VALID:
      self._valid = True

  def reset(self):
    self.theta = np.array([[KF_DEFAULT], [CDRAG_DEFAULT]], dtype=np.float64)
    self.P = np.eye(2) * 10.0
    self.points = 0
    self._valid = False

  def get_params(self):
    return float(self.theta[0, 0]), float(self.theta[1, 0])

  def is_valid(self):
    return self._valid
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest selfdrive/controls/tests/test_long_learned_mass_drag.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add sunnypilot/selfdrive/controls/lib/long_learned_mass_drag.py selfdrive/controls/tests/test_long_learned_mass_drag.py
git commit -m "feat(longitudinal): add RLSDynamicsEstimator for mass/drag learning"
```

---

## Task 3: Integrate into Longitudinal Planner

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/longitudinal_planner_ext.py`
- Modify: `selfdrive/controls/lib/longitudinal_planner.py`

- [ ] **Step 1: Extend LongitudinalPlannerSP with learning logic**

In `sunnypilot/selfdrive/controls/lib/longitudinal_planner_ext.py`, add to `LongitudinalPlannerSP`:

```python
from openpilot.sunnypilot.selfdrive.controls.lib.long_learned_mass_drag import RLSDynamicsEstimator
from openpilot.common.params import Params


class LongitudinalPlannerSP:
  # ... existing __init__ ...

  def _init_mass_drag(self):
    self.mass_drag_estimator = RLSDynamicsEstimator()
    self._params = Params()
    self.mass_drag_enabled = self._params.get_bool("LongLearnedMassDragToggle")
    self.last_mass_drag_write = 0

  def update_mass_drag(self, sm):
    if not self.mass_drag_enabled:
      return

    # Clean window conditions
    v_ego = sm['carState'].vEgo
    a_ego = sm['carState'].aEgo
    lead = sm['radarState'].leadOne
    roll = sm['liveParameters'].roll
    a_cmd = self.mpc.a_solution[0] if len(self.mpc.a_solution) > 0 else 0.0

    clean = (
      v_ego > 10.0 and
      not lead.status and
      abs(roll) < np.radians(1.0) and
      abs(a_ego - a_cmd) < 1.0
    )

    if clean:
      self.mass_drag_estimator.update(v_ego, a_cmd, a_ego)
      if self.mass_drag_estimator.is_valid():
        k_force, c_drag = self.mass_drag_estimator.get_params()
        # Write to params every 10 seconds
        if sm.logMonoTime['carState'] * 1e-9 - self.last_mass_drag_write > 10.0:
          self._params.put_nonblocking("LongLearnedKForce", str(k_force))
          self._params.put_nonblocking("LongLearnedCDrag", str(c_drag))
          self.last_mass_drag_write = sm.logMonoTime['carState'] * 1e-9
```

- [ ] **Step 2: Wire into LongitudinalPlanner.update()**

In `selfdrive/controls/lib/longitudinal_planner.py`, add at the end of `update()`:

```python
    self.update_mass_drag(sm)
```

- [ ] **Step 3: Commit**

```bash
git add sunnypilot/selfdrive/controls/lib/longitudinal_planner_ext.py selfdrive/controls/lib/longitudinal_planner.py
git commit -m "feat(longitudinal_planner): integrate mass/drag RLS learning"
```

---

## Task 4: Consume Learned Params in Longcontrol

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`
- Modify: `selfdrive/controls/lib/longcontrol.py`

- [ ] **Step 1: Add extension hook in longcontrol.py**

In `selfdrive/controls/lib/longcontrol.py`, modify `LongControl.__init__`:

```python
    self.extension = LongControlExt(self, CP, CP_SP)
```

Modify `LongControl.update()`, after computing `output_accel` in PID state:

```python
    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target)
      output_accel = self.extension.adjust_output(output_accel, CS, a_target)
```

- [ ] **Step 2: Implement LongControlExt**

Create or modify `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`:

```python
from openpilot.common.params import Params


class LongControlExt:
  def __init__(self, longcontrol, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self._params = Params()
    self.mass_drag_enabled = self._params.get_bool("LongLearnedMassDragToggle")
    self.k_force = 1.0
    self.c_drag = 0.0
    self._last_read = 0

  def _refresh_params(self, t):
    if t - self._last_read > 3.0:
      self.k_force = float(self._params.get("LongLearnedKForce", return_default=True))
      self.c_drag = float(self._params.get("LongLearnedCDrag", return_default=True))
      self._last_read = t

  def adjust_output(self, output_accel, CS, a_target):
    if not self.mass_drag_enabled:
      return output_accel
    self._refresh_params(CS.vEgo)  # t not available, use vEgo as proxy or pass t

    # Compensate feedforward for learned mass/drag
    # a_target = k_force * a_cmd - c_drag * v^2
    # Solve for a_cmd: a_cmd = (a_target + c_drag * v^2) / k_force
    if self.k_force > 0.1:
      drag_term = self.c_drag * CS.vEgo ** 2
      output_accel = (output_accel + drag_term) / self.k_force

    return float(np.clip(output_accel, self.CP.maxBrake, self.CP.maxAccel))
```

Wait, `longcontrol.py` doesn't have access to `t` (time). We can use monotonic time or just read params every call since Params.get is cached. Let me simplify:

```python
  def adjust_output(self, output_accel, CS, a_target):
    if not self.mass_drag_enabled:
      return output_accel
    self.k_force = float(self._params.get("LongLearnedKForce", return_default=True))
    self.c_drag = float(self._params.get("LongLearnedCDrag", return_default=True))

    if 0.5 <= self.k_force <= 2.0 and self.k_force != 1.0:
      drag_term = self.c_drag * CS.vEgo ** 2
      output_accel = (output_accel + drag_term) / self.k_force

    return float(np.clip(output_accel, ACCEL_MIN, ACCEL_MAX))
```

- [ ] **Step 3: Commit**

```bash
git add selfdrive/controls/lib/longcontrol.py sunnypilot/selfdrive/controls/lib/longcontrol_ext.py
git commit -m "feat(longcontrol): apply learned mass/drag compensation to output accel"
```

---

## Task 5: Integration Testing

- [ ] **Step 1: Run unit tests**

```bash
cd /home/jjolano/Developer/projects/sunnypilot
pytest selfdrive/controls/tests/test_long_learned_mass_drag.py -v
```

Expected: PASS

- [ ] **Step 2: Run planner/controlsd tests**

```bash
pytest selfdrive/controls/tests/test_longcontrol.py -v
pytest selfdrive/controls/tests/test_longitudinal_planner.py -v  # if exists
```

Expected: Existing tests still pass

- [ ] **Step 3: Commit any fixes**

```bash
git add -A && git commit -m "test: validate mass/drag learning integration"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** RLS estimator, clean windows, feedforward compensation — all covered
- [ ] **Placeholder scan:** No TBDs
- [ ] **Type consistency:** `RLSDynamicsEstimator.get_params()` returns `(float, float)` throughout
- [ ] **Safety:** Sanity clamping and reset on NaN/out-of-bounds
