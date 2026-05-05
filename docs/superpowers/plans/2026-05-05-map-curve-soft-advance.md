# Map Curve Soft Advance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-backed intermediate SCC-map target so strong map curve slowdowns start earlier without blindly trusting unconfirmed map data.

**Architecture:** Keep the existing full map-target confirmation path. When a large map slowdown is rejected only because the model does not yet confirm the full low speed, derive a bounded intermediate target from the model curve prediction and publish that through the existing SCC-map `vTarget` path. Tests lock the no-model, intermediate-model, advisory-intermediate, full-model, full-target-release, and release behaviors.

**Tech Stack:** Python, pytest, cereal custom message enums, sunnypilot `SmartCruiseControlMap`.

---

## File Structure

- Modify: `sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py`
- Modify: `sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py`
- Reference: `docs/superpowers/specs/2026-05-05-map-curve-soft-advance-design.md`

## Baseline

The retained branch worktree is `.worktrees/longitudinal-osm-planner` on `feat/longitudinal-osm-planner`.

Baseline command already passed before this plan was written:

```bash
uv run --extra testing --extra tools python -m pytest sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py
```

Expected baseline: `23 passed`.

### Task 1: Add Failing Soft-Advance Tests

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py`

- [ ] **Step 1: Add a test for model-backed intermediate target selection**

Insert this test after `test_target_velocity_ignores_relative_slowdown_without_model_curve`:

```python
  def test_target_velocity_uses_model_intermediate_when_full_map_target_unconfirmed(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 13.0
    model_target_v = 18.0
    yaw_rate = self.scc_m.v_ego * map_controller.MODEL_CURVE_TARGET_LAT_ACCEL / model_target_v**2
    distance = self.scc_m._target_control_distance(model_target_v) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance=distance, yaw_rate=yaw_rate, speed=self.scc_m.v_ego)
    expected_target = self.scc_m._prediction_curve_target(model_msg, distance)

    assert expected_target is not None
    assert target_v < expected_target < self.scc_m.v_ego
    assert expected_target > target_v + map_controller.MODEL_CURVE_OVERSLOWDOWN_MARGIN

    for _ in range(2):
      self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, model_msg)

    assert self.scc_m.state == VisionState.turning
    assert self.scc_m.output_v_target == pytest.approx(expected_target)
```

- [ ] **Step 2: Add a release test for intermediate targets**

Insert this test after the previous one:

```python
  def test_model_intermediate_target_releases_when_prediction_drops(self):
    self.scc_m.v_ego = 25.0
    self.scc_m.a_ego = 0.0
    target_v = 13.0
    model_target_v = 18.0
    yaw_rate = self.scc_m.v_ego * map_controller.MODEL_CURVE_TARGET_LAT_ACCEL / model_target_v**2
    distance = self.scc_m._target_control_distance(model_target_v) - 1.0
    target_lon = distance / R * TO_DEGREES
    self.mem_params.put("MapTargetVelocities", json.dumps([
      {"latitude": 0.0, "longitude": target_lon, "velocity": target_v},
    ]))
    model_msg = make_model_prediction(distance=distance, yaw_rate=yaw_rate, speed=self.scc_m.v_ego)

    for _ in range(2):
      self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, model_msg)
    assert self.scc_m.state == VisionState.turning
    assert target_v < self.scc_m.output_v_target < self.scc_m.v_ego

    weak_model_msg = make_model_prediction(distance=distance, yaw_rate=0.02, speed=self.scc_m.v_ego)
    self.scc_m.update(True, False, self.scc_m.v_ego, 0.0, 30.0, weak_model_msg)

    assert self.scc_m.state == VisionState.enabled
    assert self.scc_m.output_v_target == V_CRUISE_UNSET
```

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```bash
uv run --extra testing --extra tools python -m pytest \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py::TestSmartCruiseControlMap::test_target_velocity_uses_model_intermediate_when_full_map_target_unconfirmed \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py::TestSmartCruiseControlMap::test_model_intermediate_target_releases_when_prediction_drops -q
```

Expected: first test fails because `output_v_target` is `V_CRUISE_UNSET` or the full map target is not replaced by the intermediate model target. The second test may fail at the first active-state assertion for the same reason.

### Task 2: Implement Soft Model-Backed Map Advance

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py`

- [ ] **Step 1: Add a helper for intermediate model targets**

Add this method after `_model_confirms_large_slowdown(...)`:

```python
  @classmethod
  def _soft_model_confirmed_target(cls, v_ego: float, target_v: float, distance: float, model_msg) -> float | None:
    if model_msg is None or not cls._model_covers_distance(model_msg, distance):
      return None

    prediction_target = cls._prediction_curve_target(model_msg, distance)
    if prediction_target is None:
      return None

    soft_target = max(target_v, min(v_ego, prediction_target))
    return soft_target if soft_target < v_ego else None
```

- [ ] **Step 2: Change `_target_range_state` to return the selected target speed**

Replace `_target_range_state(...)` with:

```python
  def _target_range_state(self, target_v: float, distance: float, model_msg) -> tuple[bool, bool, float]:
    if not self._model_confirms_large_slowdown(self.v_ego, target_v, distance, model_msg):
      soft_target = self._soft_model_confirmed_target(self.v_ego, target_v, distance, model_msg)
      if soft_target is not None and self._target_in_range(soft_target, distance):
        return True, True, soft_target
      return False, False, target_v

    if self._target_in_range(target_v, distance):
      return True, False, target_v

    control_target_v = self._prediction_control_target(target_v, distance, model_msg)
    prediction_advanced = control_target_v < target_v and self._target_in_range(control_target_v, distance)
    return prediction_advanced, prediction_advanced, target_v
```

- [ ] **Step 3: Update forward target velocity caller**

In `update_calculations(...)`, replace:

```python
      in_range, prediction_advanced = self._target_range_state(tv, d, model_msg)
      if in_range:
        valid_velocities.append((float(tv), tlat, tlon, prediction_advanced))
```

with:

```python
      in_range, prediction_advanced, control_tv = self._target_range_state(tv, d, model_msg)
      if in_range:
        valid_velocities.append((float(control_tv), tlat, tlon, prediction_advanced))
```

- [ ] **Step 4: Update advisory target callers**

In `_advisory_targets(...)`, replace the current-advisory block:

```python
      in_range, prediction_advanced = self._target_range_state(current_target[0], 0., model_msg)
      if in_range:
        targets.append((*current_target, prediction_advanced))
```

with:

```python
      in_range, prediction_advanced, control_target_v = self._target_range_state(current_target[0], 0., model_msg)
      if in_range:
        targets.append((control_target_v, current_target[1], current_target[2], prediction_advanced))
```

Replace the next-advisory block:

```python
      in_range, prediction_advanced = self._target_range_state(next_target[0], next_distance, model_msg)
      if in_range:
        targets.append((*next_target, prediction_advanced))
```

with:

```python
      in_range, prediction_advanced, control_target_v = self._target_range_state(next_target[0], next_distance, model_msg)
      if in_range:
        targets.append((control_target_v, next_target[1], next_target[2], prediction_advanced))
```

- [ ] **Step 5: Run the focused tests**

Run:

```bash
uv run --extra testing --extra tools python -m pytest \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py::TestSmartCruiseControlMap::test_target_velocity_uses_model_intermediate_when_full_map_target_unconfirmed \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py::TestSmartCruiseControlMap::test_model_intermediate_target_releases_when_prediction_drops -q
```

Expected: both tests pass.

### Task 3: Protect Existing Map Behavior

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py`
- Modify: `sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py` only if these regressions fail

- [ ] **Step 1: Run the existing no-model and full-model tests**

Run:

```bash
uv run --extra testing --extra tools python -m pytest \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py::TestSmartCruiseControlMap::test_target_velocity_ignores_large_slowdown_without_model_curve \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py::TestSmartCruiseControlMap::test_current_advisory_speed_limit_allows_model_confirmed_curve \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py::TestSmartCruiseControlMap::test_model_advanced_map_target_releases_when_prediction_drops -q
```

Expected: all selected tests pass.

- [ ] **Step 2: If `test_model_advanced_map_target_releases_when_prediction_drops` fails, keep existing release semantics**

The intended release condition is unchanged: when model evidence disappears and the target was prediction-advanced, `update_calculations(...)` must clear the target and `update(...)` must return to `MapState.enabled` with `V_CRUISE_UNSET`.

If needed, keep this reset block unchanged:

```python
    if self.v_target < min_v and not (self.target_lat == 0 and self.target_lon == 0):
      if not self.target_prediction_advanced:
        for i in range(len(forward_points)):
          target_velocity = forward_points[i]
          tlat = target_velocity["latitude"]
          tlon = target_velocity["longitude"]
          tv = float(target_velocity["velocity"])
          if tv > self.v_ego:
            continue

          if tlat == self.target_lat and tlon == self.target_lon and tv == self.v_target:
            return

      # not found so let's reset
      self._clear_map_target()
```

### Task 4: Run Full Map Controller Regression

**Files:**
- Test: `sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py`

- [ ] **Step 1: Run the full map-controller test file**

Run:

```bash
uv run --extra testing --extra tools python -m pytest sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py
```

Expected: all tests pass, with the count increasing from `23 passed` to `27 passed`.

- [ ] **Step 2: Inspect the diff**

Run:

```bash
git diff -- sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py docs/superpowers/specs/2026-05-05-map-curve-soft-advance-design.md docs/superpowers/plans/2026-05-05-map-curve-soft-advance.md
```

Expected: diff only contains the soft-advance helper, `_target_range_state` return-shape/caller updates, the preservation-block confirmation check, the four tests, and the spec/plan docs.

### Task 5: Completion Gate

**Files:**
- Review: working tree status

- [ ] **Step 1: Check retained worktree status**

Run:

```bash
git status --short
```

Expected: only these files are modified or untracked:

```text
M  sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py
M  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py
?? docs/superpowers/specs/2026-05-05-map-curve-soft-advance-design.md
?? docs/superpowers/plans/2026-05-05-map-curve-soft-advance.md
```

- [ ] **Step 2: Commit only if explicitly requested**

This repository requires explicit user approval before commits. If the user has requested a commit, run:

```bash
git add sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py \
  sunnypilot/selfdrive/controls/lib/smart_cruise_control/tests/test_map_controller.py \
  docs/superpowers/specs/2026-05-05-map-curve-soft-advance-design.md \
  docs/superpowers/plans/2026-05-05-map-curve-soft-advance.md
git commit -m "controls: softly advance map curve targets"
```

Expected if committed: a new commit on `feat/longitudinal-osm-planner`. If no commit was explicitly requested, do not run the commit command.
