# Brake/Gas Response Curve Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn the non-linear mapping from commanded acceleration to actual vehicle acceleration, bucketed by accel magnitude. Use learned offsets to improve feedforward accuracy in the longitudinal PID controller.

**Architecture:** Extend `longcontrol.py` via `longcontrol_ext.py` to maintain per-bucket mean offsets between `a_cmd` and `a_ego`. Offsets are smoothed with first-order filters and applied as corrections to the feedforward term.

**Tech Stack:** Python 3, NumPy, openpilot Params, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py` | Response curve learner: bucketing, filtering, offset lookup |
| `selfdrive/controls/lib/longcontrol.py` | Wires learner into PID update loop |
| `selfdrive/controls/tests/test_response_curve_learner.py` | Unit tests for bucketing and interpolation |
| `common/params_keys.h` | New param key definitions |
| `sunnypilot/sunnylink/params_metadata.json` | Param descriptions |

---

## Prerequisites

Ensure you have read:
- `docs/superpowers/specs/2026-05-04-live-learning-expansion-design.md` Section 5
- `selfdrive/controls/lib/longcontrol.py` — understand `LongControl.update()` and PID feedforward
- `selfdrive/controls/lib/longcontrol_ext.py` — understand extension pattern from Plan 2

---

## Task 1: Add Param Keys and Metadata

**Files:**
- Modify: `common/params_keys.h`
- Modify: `sunnypilot/sunnylink/params_metadata.json`

- [ ] **Step 1: Add param keys**

In `common/params_keys.h`, add:

```c
    {"LongLearnedResponseCurveToggle", {PERSISTENT | BACKUP, BOOL, "0"}},
    {"LongLearnedResponseOffsets", {PERSISTENT, TEXT, ""}},
```

- [ ] **Step 2: Add metadata**

In `sunnypilot/sunnylink/params_metadata.json`:

```json
  "LongLearnedResponseCurveToggle": {
    "title": "Learn Response Curve",
    "description": "Learns the non-linear relationship between requested and actual acceleration to improve pedal response."
  },
  "LongLearnedResponseOffsets": {
    "title": "Learned Response Offsets",
    "description": ""
  },
```

- [ ] **Step 3: Commit**

```bash
git add common/params_keys.h sunnypilot/sunnylink/params_metadata.json
git commit -m "params: add LongLearnedResponseCurveToggle and LongLearnedResponseOffsets"
```

---

## Task 2: Implement Response Curve Learner

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`
- Create: `selfdrive/controls/tests/test_response_curve_learner.py`

- [ ] **Step 1: Write the failing test**

Create `selfdrive/controls/tests/test_response_curve_learner.py`:

```python
#!/usr/bin/env python3
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.longcontrol_ext import ResponseCurveLearner


def test_bucket_routing():
  learner = ResponseCurveLearner()
  assert learner._bucket_idx(-3.0) == 0
  assert learner._bucket_idx(-0.3) == 3
  assert learner._bucket_idx(0.0) == 4
  assert learner._bucket_idx(0.3) == 4
  assert learner._bucket_idx(1.5) == 6
  assert learner._bucket_idx(5.0) == 8


def test_update_and_lookup():
  learner = ResponseCurveLearner()
  # Bucket 6 is [1.0, 2.0); add points with offset +0.3
  for _ in range(20):
    learner.update(1.5, 1.5 + 0.3)

  assert learner.is_bucket_valid(6)
  offset = learner.lookup_offset(1.5)
  assert abs(offset - 0.3) < 0.05


def test_interpolation():
  learner = ResponseCurveLearner()
  # Bucket 5: [0.5, 1.0) -> offset -0.2
  for _ in range(20):
    learner.update(0.7, 0.7 - 0.2)
  # Bucket 7: [2.0, 4.0) -> offset +0.4
  for _ in range(20):
    learner.update(3.0, 3.0 + 0.4)

  # Interpolate between 0.7 and 3.0 at 1.5
  offset = learner.lookup_offset(1.5)
  expected = np.interp(1.5, [0.7, 3.0], [-0.2, 0.4])
  assert abs(offset - expected) < 0.1


def test_sanity_clamp():
  learner = ResponseCurveLearner()
  for _ in range(20):
    learner.update(1.0, 10.0)  # impossible offset

  offset = learner.lookup_offset(1.0)
  assert offset <= 0.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/jjolano/Developer/projects/sunnypilot
pytest selfdrive/controls/tests/test_response_curve_learner.py -v
```

Expected: FAIL with `ImportError: cannot import name 'ResponseCurveLearner'`

- [ ] **Step 3: Implement ResponseCurveLearner**

In `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`, add:

```python
import numpy as np
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_CTRL

ACCEL_BUCKET_BP = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
MIN_BUCKET_SAMPLES = 10
MAX_OFFSET = 0.5
FILTER_DECAY = 100


class ResponseCurveLearner:
  def __init__(self):
    self.buckets = {i: [] for i in range(len(ACCEL_BUCKET_BP))}
    self.filtered_offsets = {i: FirstOrderFilter(0.0, FILTER_DECAY, DT_CTRL) for i in range(len(ACCEL_BUCKET_BP))}
    self._valid = {i: False for i in range(len(ACCEL_BUCKET_BP))}

  def _bucket_idx(self, a_cmd):
    for i in range(len(ACCEL_BUCKET_BP) - 1):
      if ACCEL_BUCKET_BP[i] <= a_cmd < ACCEL_BUCKET_BP[i + 1]:
        return i
    return len(ACCEL_BUCKET_BP) - 1

  def update(self, a_cmd, a_ego):
    idx = self._bucket_idx(a_cmd)
    offset = a_ego - a_cmd
    if not np.isfinite(offset):
      return
    self.buckets[idx].append(offset)
    if len(self.buckets[idx]) > 100:
      self.buckets[idx].pop(0)

    if len(self.buckets[idx]) >= MIN_BUCKET_SAMPLES:
      mean_offset = float(np.mean(self.buckets[idx]))
      mean_offset = np.clip(mean_offset, -MAX_OFFSET, MAX_OFFSET)
      self.filtered_offsets[idx].update(mean_offset)
      self._valid[idx] = True

  def is_bucket_valid(self, idx):
    return self._valid.get(idx, False)

  def lookup_offset(self, a_cmd):
    idx = self._bucket_idx(a_cmd)
    if self._valid[idx]:
      return float(self.filtered_offsets[idx].x)

    # Find nearest valid buckets for interpolation
    valid_indices = [i for i in range(len(ACCEL_BUCKET_BP)) if self._valid[i]]
    if not valid_indices:
      return 0.0
    if len(valid_indices) == 1:
      return float(self.filtered_offsets[valid_indices[0]].x)

    # Interpolate based on bucket center points
    centers = []
    offsets = []
    for i in valid_indices:
      center = ACCEL_BUCKET_BP[i] if i < len(ACCEL_BUCKET_BP) - 1 else ACCEL_BUCKET_BP[-1]
      centers.append(center)
      offsets.append(float(self.filtered_offsets[i].x))

    return float(np.interp(a_cmd, centers, offsets))

  def serialize(self):
    data = {}
    for i in range(len(ACCEL_BUCKET_BP)):
      if self._valid[i]:
        data[i] = float(self.filtered_offsets[i].x)
    return str(data)

  def deserialize(self, s):
    if not s:
      return
    try:
      import ast
      data = ast.literal_eval(s)
      for i, val in data.items():
        i = int(i)
        if 0 <= i < len(ACCEL_BUCKET_BP):
          self.filtered_offsets[i].x = float(val)
          self._valid[i] = True
    except Exception:
      pass
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest selfdrive/controls/tests/test_response_curve_learner.py -v
```

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add sunnypilot/selfdrive/controls/lib/longcontrol_ext.py selfdrive/controls/tests/test_response_curve_learner.py
git commit -m "feat(longcontrol_ext): add ResponseCurveLearner for brake/gas response"
```

---

## Task 3: Integrate into LongControl

**Files:**
- Modify: `selfdrive/controls/lib/longcontrol.py`
- Modify: `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`

- [ ] **Step 1: Wire learner into PID state**

In `selfdrive/controls/lib/longcontrol.py`, in `LongControl.update()`:

```python
    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      # Apply learned response curve correction to feedforward
      ff = a_target + self.extension.get_response_offset(a_target)
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=ff)
      output_accel = self.extension.adjust_output(output_accel, CS, a_target)
```

- [ ] **Step 2: Add response methods to LongControlExt**

In `sunnypilot/selfdrive/controls/lib/longcontrol_ext.py`, add to `LongControlExt`:

```python
  def __init__(self, longcontrol, CP, CP_SP):
    # ... existing init from Plan 2 ...
    self.response_curve_enabled = self._params.get_bool("LongLearnedResponseCurveToggle")
    self.response_learner = ResponseCurveLearner()
    # Restore cached offsets
    cached = self._params.get("LongLearnedResponseOffsets")
    if cached:
      self.response_learner.deserialize(cached)

  def get_response_offset(self, a_target):
    if not self.response_curve_enabled:
      return 0.0
    return self.response_learner.lookup_offset(a_target)

  def learn_response(self, a_target, a_ego, long_control_state, saturated):
    if not self.response_curve_enabled:
      return
    if long_control_state != LongCtrlState.pid:
      return
    if saturated:
      return
    if abs(a_ego - a_target) >= 1.0:
      return
    self.response_learner.update(a_target, a_ego)

  def cache_response_params(self):
    if not self.response_curve_enabled:
      return
    self._params.put_nonblocking("LongLearnedResponseOffsets", self.response_learner.serialize())
```

- [ ] **Step 3: Call learning in LongControl.update()**

In `selfdrive/controls/lib/longcontrol.py`, after computing output_accel in PID state:

```python
    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      ff = a_target + self.extension.get_response_offset(a_target)
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=ff)
      self.extension.learn_response(a_target, CS.aEgo, self.long_control_state,
                                    self.pid.saturated if hasattr(self.pid, 'saturated') else False)
      output_accel = self.extension.adjust_output(output_accel, CS, a_target)
```

Add periodic caching. In `__init__` add `self._response_cache_frame = 0`, and in `update()`:

```python
    self._response_cache_frame += 1
    if self._response_cache_frame % 600 == 0:  # every ~6 seconds at 100Hz
      self.extension.cache_response_params()
```

Wait, `LongControl` runs at 100Hz (DT_CTRL = 0.01). 600 frames = 6 seconds. But we're mixing concerns. Let's just add a simple time-based cache in the extension or call it from controlsd. Actually, simpler: call `cache_response_params()` in the extension's `learn_response()` every N calls, or use a timer.

Simpler approach: In `longcontrol_ext.py`:

```python
  def learn_response(self, a_target, a_ego, long_control_state, saturated):
    # ... existing validation ...
    self.response_learner.update(a_target, a_ego)
    self._response_learn_count += 1
    if self._response_learn_count % 500 == 0:
      self._params.put_nonblocking("LongLearnedResponseOffsets", self.response_learner.serialize())
```

- [ ] **Step 4: Commit**

```bash
git add selfdrive/controls/lib/longcontrol.py sunnypilot/selfdrive/controls/lib/longcontrol_ext.py
git commit -m "feat(longcontrol): integrate response curve learning into PID feedforward"
```

---

## Task 4: Integration Testing

- [ ] **Step 1: Run unit tests**

```bash
cd /home/jjolano/Developer/projects/sunnypilot
pytest selfdrive/controls/tests/test_response_curve_learner.py -v
pytest selfdrive/controls/tests/test_longcontrol.py -v
```

Expected: PASS

- [ ] **Step 2: Validate no regressions in process replay**

```bash
python selfdrive/test/process_replay/process_replay.py selfdrive/controls/lib/longcontrol.py
```

Expected: No crashes

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: validate response curve learning integration"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Bucketing, filtering, interpolation, feedforward correction — all covered
- [ ] **Placeholder scan:** No TBDs
- [ ] **Type consistency:** `lookup_offset` returns `float` everywhere
- [ ] **Safety:** Offsets clamped to `±0.5`, invalid buckets return 0.0
