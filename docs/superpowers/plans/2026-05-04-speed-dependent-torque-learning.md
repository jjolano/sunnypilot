# Speed-Dependent Torque Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live learning of speed-dependent `latAccelFactor` values for lateral torque control, so the controller uses different steering gains at different vehicle speeds.

**Architecture:** Extend `torqued.py` via `torqued_ext.py` to maintain separate torque parameter buckets per speed band. Publish learned bucket values via Params. `latcontrol_torque_ext.py` interpolates the correct factor based on current `vEgo`.

**Tech Stack:** Python 3, NumPy, openpilot cereal messaging, openpilot Params, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `sunnypilot/selfdrive/locationd/torqued_ext.py` | Extension mixin: speed bucket logic, param serialization |
| `selfdrive/locationd/torqued.py` | Main process: wires `vEgo` into point addition |
| `sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py` | Consumes speed-aware params, interpolates factor |
| `sunnypilot/selfdrive/controls/controlsd_ext.py` | Reads speed-aware param and passes to lateral controller |
| `selfdrive/locationd/test/test_torqued_speed_adaptive.py` | Unit tests for speed bucket estimation |
| `common/params_keys.h` | New param key definitions |
| `sunnypilot/sunnylink/params_metadata.json` | Param descriptions for UI sync |
| `selfdrive/ui/sunnypilot/layouts/settings/steering_sub_layouts/torque_settings.py` | Settings toggle |

---

## Prerequisites

Ensure you have read:
- `docs/superpowers/specs/2026-05-04-live-learning-expansion-design.md` Section 3
- `selfdrive/locationd/torqued.py` — understand `TorqueEstimator`, `TorqueBuckets`, `get_msg()`
- `selfdrive/locationd/helpers.py` — understand `PointBuckets`
- `sunnypilot/selfdrive/locationd/torqued_ext.py` — understand current extension hooks
- `sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py` — understand `update_override_torque_params`

---

## Task 1: Add Param Keys and Metadata

**Files:**
- Modify: `common/params_keys.h`
- Modify: `sunnypilot/sunnylink/params_metadata.json`

- [ ] **Step 1: Add param keys**

In `common/params_keys.h`, add after the existing torque params (around line 230):

```c
    {"LiveTorqueSpeedAdaptiveToggle", {PERSISTENT | BACKUP, BOOL, "0"}},
    {"LiveTorqueSpeedAdaptiveParams", {PERSISTENT, TEXT, ""}},
```

- [ ] **Step 2: Add metadata**

In `sunnypilot/sunnylink/params_metadata.json`, add after `LiveTorqueParamsToggle`:

```json
  "LiveTorqueSpeedAdaptiveToggle": {
    "title": "Speed-Adaptive Self-Tune",
    "description": "Learn separate torque parameters for different speed ranges. Improves steering accuracy across low and high speeds."
  },
  "LiveTorqueSpeedAdaptiveParams": {
    "title": "Speed-Adaptive Torque Parameters",
    "description": ""
  },
```

- [ ] **Step 3: Commit**

```bash
git add common/params_keys.h sunnypilot/sunnylink/params_metadata.json
git commit -m "params: add LiveTorqueSpeedAdaptiveToggle and LiveTorqueSpeedAdaptiveParams"
```

---

## Task 2: Implement Speed-Aware Torque Buckets

**Files:**
- Modify: `sunnypilot/selfdrive/locationd/torqued_ext.py`
- Create: `selfdrive/locationd/test/test_torqued_speed_adaptive.py`

- [ ] **Step 1: Write the failing test**

Create `selfdrive/locationd/test/test_torqued_speed_adaptive.py`:

```python
#!/usr/bin/env python3
import numpy as np
import pytest

from openpilot.selfdrive.locationd.helpers import PointBuckets
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import SpeedAwareTorqueBuckets


def test_speed_aware_buckets_routing():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([10, 50, 10]),
    min_points_total=20,
    points_per_bucket=100,
    rowsize=3
  )
  buckets.add_point(-0.4, 1.0, 5.0)
  buckets.add_point(0.4, 2.0, 15.0)
  buckets.add_point(0.0, 3.0, 25.0)

  assert len(buckets.buckets_for_speed(5.0).buckets) == 3
  assert len(buckets.buckets_for_speed(5.0).get_points()) == 1
  assert len(buckets.buckets_for_speed(15.0).get_points()) == 1
  assert len(buckets.buckets_for_speed(25.0).get_points()) == 1


def test_speed_aware_buckets_valid_percent():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for _ in range(10):
    buckets.add_point(0.0, 1.0, 5.0)
    buckets.add_point(0.0, 1.0, 15.0)

  assert buckets.is_calculable()
  assert buckets.is_valid()


def test_speed_aware_get_points():
  buckets = SpeedAwareTorqueBuckets(
    x_bounds=[(-0.5, -0.3), (-0.3, 0.3), (0.3, 0.5)],
    speed_bp=[0, 10, 20],
    min_points=np.array([1, 1, 1]),
    min_points_total=2,
    points_per_bucket=100,
    rowsize=3
  )
  for i in range(5):
    buckets.add_point(float(i) * 0.1, float(i), 5.0)

  pts = buckets.get_points(10)
  assert len(pts) == 5
  assert pts.shape[1] == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/jjolano/Developer/projects/sunnypilot
pytest selfdrive/locationd/test/test_torqued_speed_adaptive.py -v
```

Expected: FAIL with `ImportError: cannot import name 'SpeedAwareTorqueBuckets'`

- [ ] **Step 3: Implement SpeedAwareTorqueBuckets**

In `sunnypilot/selfdrive/locationd/torqued_ext.py`, add:

```python
import numpy as np
from openpilot.selfdrive.locationd.helpers import PointBuckets


class SpeedAwareTorqueBuckets:
  def __init__(self, x_bounds, speed_bp, min_points, min_points_total, points_per_bucket, rowsize=3):
    self.x_bounds = x_bounds
    self.speed_bp = list(speed_bp)
    self.min_points = min_points
    self.min_points_total = min_points_total
    self.points_per_bucket = points_per_bucket
    self.rowsize = rowsize
    self.buckets = {}
    self._init_buckets()

  def _init_buckets(self):
    for i in range(len(self.speed_bp)):
      self.buckets[i] = PointBuckets(
        x_bounds=self.x_bounds,
        min_points=self.min_points,
        min_points_total=self.min_points_total,
        points_per_bucket=self.points_per_bucket,
        rowsize=self.rowsize
      )

  def _bucket_idx(self, v_ego):
    for i in range(len(self.speed_bp) - 1):
      if self.speed_bp[i] <= v_ego < self.speed_bp[i + 1]:
        return i
    return len(self.speed_bp) - 1

  def add_point(self, x, y, v_ego):
    idx = self._bucket_idx(v_ego)
    self.buckets[idx].add_point(x, y)

  def buckets_for_speed(self, v_ego):
    return self.buckets[self._bucket_idx(v_ego)]

  def is_calculable(self):
    return any(b.is_calculable() for b in self.buckets.values())

  def is_valid(self):
    return any(b.is_valid() for b in self.buckets.values())

  def get_points(self, n=None):
    all_pts = []
    for b in self.buckets.values():
      pts = b.get_points(n)
      if len(pts) > 0:
        all_pts.append(pts)
    if not all_pts:
      return np.empty((0, self.rowsize))
    return np.vstack(all_pts)

  def get_valid_percent(self):
    vals = [b.get_valid_percent() for b in self.buckets.values() if b.is_calculable()]
    return float(np.mean(vals)) if vals else 0.0

  def total_points(self):
    return sum(len(b) for b in self.buckets.values())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest selfdrive/locationd/test/test_torqued_speed_adaptive.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add sunnypilot/selfdrive/locationd/torqued_ext.py selfdrive/locationd/test/test_torqued_speed_adaptive.py
git commit -m "feat(torqued_ext): add SpeedAwareTorqueBuckets for speed-dependent torque learning"
```

---

## Task 3: Integrate Speed Buckets into torqued.py

**Files:**
- Modify: `selfdrive/locationd/torqued.py`
- Modify: `sunnypilot/selfdrive/locationd/torqued_ext.py`

- [ ] **Step 1: Add extension hooks in torqued.py**

In `selfdrive/locationd/torqued.py`, modify `TorqueEstimator`:

Add method after `__init__`:
```python
  def add_filtered_point(self, steer, lateral_acc, v_ego):
    self.filtered_points.add_point(steer, lateral_acc)
    TorqueEstimatorExt.add_speed_aware_point(self, steer, lateral_acc, v_ego)
```

In `handle_log`, change:
```python
          if abs(lateral_acc) <= LAT_ACC_THRESHOLD:
            self.filtered_points.add_point(steer, lateral_acc)
```
to:
```python
          if abs(lateral_acc) <= LAT_ACC_THRESHOLD:
            self.add_filtered_point(steer, lateral_acc, vego)
```

- [ ] **Step 2: Add speed-aware estimation in torqued_ext.py**

Add to `sunnypilot/selfdrive/locationd/torqued_ext.py`:

```python
SPEED_BUCKET_BP = [0, 10, 20, 30, 40]  # m/s
SPEED_BUCKET_LABELS = ["0_10", "10_20", "20_30", "30_40", "40_plus"]


class TorqueEstimatorExt:
  # ... existing __init__ ...

  def initialize_custom_params(self, decimated=False):
    # ... existing code ...
    self.speed_adaptive_enabled = self._params.get_bool("LiveTorqueSpeedAdaptiveToggle")
    self.speed_bucket_params = {}
    self.speed_bucket_filters = {}
    self._init_speed_buckets()

  def _init_speed_buckets(self):
    from openpilot.selfdrive.locationd.helpers import PointBuckets
    min_pts = RELAXED_MIN_BUCKET_POINTS / (10 if hasattr(self, 'decimated') and self.decimated else 1)
    self.speed_buckets = SpeedAwareTorqueBuckets(
      x_bounds=STEER_BUCKET_BOUNDS,
      speed_bp=SPEED_BUCKET_BP,
      min_points=min_pts,
      min_points_total=self.min_points_total,
      points_per_bucket=POINTS_PER_BUCKET,
      rowsize=3
    )

  def add_speed_aware_point(self, steer, lateral_acc, v_ego):
    if not self.speed_adaptive_enabled:
      return
    self.speed_buckets.add_point(steer, lateral_acc, v_ego)

  def estimate_speed_aware_params(self):
    """Returns dict of bucket_label -> (latAccelFactor, latAccelOffset, frictionCoeff)"""
    from openpilot.selfdrive.locationd.torqued import slope2rot
    result = {}
    for idx, label in enumerate(SPEED_BUCKET_LABELS):
      bucket = self.speed_buckets.buckets[idx]
      if not bucket.is_calculable():
        continue
      points = bucket.get_points(self.fit_points)
      try:
        _, _, v = np.linalg.svd(points, full_matrices=False)
        slope, offset = -v.T[0:2, 2] / v.T[2, 2]
        _, spread = np.matmul(points[:, [0, 2]], slope2rot(slope)).T
        friction_coeff = np.std(spread) * FRICTION_FACTOR
        result[label] = (float(slope), float(offset), float(friction_coeff))
      except np.linalg.LinAlgError:
        continue
    return result
```

- [ ] **Step 3: Serialize and cache speed-aware params in torqued.py**

In `selfdrive/locationd/torqued.py`, in `get_msg`, after the existing `liveValid` block, add:

```python
    # Speed-aware params
    if self.speed_adaptive_enabled and self.speed_buckets.is_calculable():
      speed_params = self.estimate_speed_aware_params()
      if speed_params:
        liveTorqueParameters.speedAdaptiveValid = True
        liveTorqueParameters.speedAdaptiveParams = str(speed_params)
      else:
        liveTorqueParameters.speedAdaptiveValid = False
```

Wait, we can't add new fields to `LiveTorqueParametersData` without cereal changes. Instead, write to Params directly in `main()`:

In `selfdrive/locationd/torqued.py` `main()`, inside the loop after `pm.send`:

```python
    if estimator.speed_adaptive_enabled and sm.frame % 240 == 0:
      speed_params = estimator.estimate_speed_aware_params()
      if speed_params:
        params.put_nonblocking("LiveTorqueSpeedAdaptiveParams", str(speed_params))
```

- [ ] **Step 4: Commit**

```bash
git add selfdrive/locationd/torqued.py sunnypilot/selfdrive/locationd/torqued_ext.py
git commit -m "feat(torqued): integrate speed-aware torque buckets and param caching"
```

---

## Task 4: Consume Speed-Aware Params in Lateral Controller

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py`
- Modify: `sunnypilot/selfdrive/controls/controlsd_ext.py`

- [ ] **Step 1: Add speed-aware interpolation in latcontrol_torque_ext.py**

In `sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py`, add:

```python
import ast


class LatControlTorqueExt:
  # ... existing __init__ ...

  def update_speed_aware_params(self, params_str):
    if not params_str:
      self.speed_aware_params = None
      return
    try:
      self.speed_aware_params = ast.literal_eval(params_str)
    except (ValueError, SyntaxError):
      self.speed_aware_params = None

  def _interpolate_speed_factor(self, v_ego):
    if not self.speed_aware_params:
      return None
    bp = [0, 10, 20, 30, 40]
    factors = []
    labels = ["0_10", "10_20", "20_30", "30_40", "40_plus"]
    for label in labels:
      if label in self.speed_aware_params:
        factors.append(self.speed_aware_params[label][0])  # latAccelFactor
      else:
        factors.append(None)

    valid = [(b, f) for b, f in zip(bp, factors) if f is not None]
    if not valid:
      return None
    if len(valid) == 1:
      return valid[0][1]
    return float(np.interp(v_ego, [b for b, _ in valid], [f for _, f in valid]))

  def update_override_torque_params(self, torque_params):
    # ... existing override logic ...
    if self.speed_aware_params is not None:
      factor = self._interpolate_speed_factor(self.last_v_ego)
      if factor is not None:
        torque_params.latAccelFactor = factor
        return True
    return False  # or existing return value
```

Add `self.last_v_ego = 0.0` in `__init__` and update it in `update()`.

- [ ] **Step 2: Wire param reading in controlsd_ext.py**

In `sunnypilot/selfdrive/controls/controlsd_ext.py`, in `get_params_sp`:

```python
      if self.CP.lateralTuning.which() == 'torque':
        self.lat_delay = get_lat_delay(self.params, sm["liveDelay"].lateralDelay)
        speed_aware_params = self.params.get("LiveTorqueSpeedAdaptiveParams")
        if hasattr(self, 'LaC') and self.LaC is not None:
          self.LaC.extension.update_speed_aware_params(speed_aware_params)
```

- [ ] **Step 3: Commit**

```bash
git add sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py sunnypilot/selfdrive/controls/controlsd_ext.py
git commit -m "feat(latcontrol): consume speed-aware torque params for interpolated latAccelFactor"
```

---

## Task 5: Add UI Toggle

**Files:**
- Modify: `selfdrive/ui/sunnypilot/layouts/settings/steering_sub_layouts/torque_settings.py`

- [ ] **Step 1: Add toggle widget**

In the `__init__` of `TorqueSettings`, after `self._relaxed_tune_toggle`, add:

```python
    self._speed_adaptive_toggle = toggle_item_sp(
      tr("Speed-Adaptive Self-Tune"),
      param="LiveTorqueSpeedAdaptiveToggle",
      description=lambda: tr("Learns separate torque parameters for different speed ranges to improve steering accuracy across all speeds."),
    )
```

Add to the layout list around line 99:
```python
      self._speed_adaptive_toggle,
```

In `update()`, add after the relaxed toggle enable logic:
```python
    self._speed_adaptive_toggle.action_item.set_enabled(ui_state.is_offroad())
```

- [ ] **Step 2: Commit**

```bash
git add selfdrive/ui/sunnypilot/layouts/settings/steering_sub_layouts/torque_settings.py
git commit -m "feat(ui): add Speed-Adaptive Self-Tune toggle in torque settings"
```

---

## Task 6: Integration Testing

- [ ] **Step 1: Run unit tests**

```bash
cd /home/jjolano/Developer/projects/sunnypilot
pytest selfdrive/locationd/test/test_torqued_speed_adaptive.py -v
pytest selfdrive/locationd/test/test_torqued.py -v
```

Expected: All tests pass

- [ ] **Step 2: Run process replay smoke test**

```bash
cd /home/jjolano/Developer/projects/sunnypilot
python selfdrive/test/process_replay/process_replay.py selfdrive/locationd/torqued.py
```

Expected: No crashes, output matches baseline within tolerance

- [ ] **Step 3: Commit any test fixes**

```bash
git add -A
git commit -m "test: add integration tests for speed-dependent torque learning"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Speed buckets, estimation, interpolation, UI toggle — all covered
- [ ] **Placeholder scan:** No TBDs or vague steps
- [ ] **Type consistency:** `SpeedAwareTorqueBuckets` API matches `PointBuckets` where applicable
- [ ] **Safety:** Learned values only applied when `is_calculable()` and `is_valid()`
