# Lateral Accel Curve-Entry Burst Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a saturation-latched temporary lateral acceleration burst up to `5.0 m/s^2` while preserving default `3.0 m/s^2` behavior and existing manual-gas override semantics.

**Architecture:** Keep the feature inside `selfdrive/controls/lib/drive_helpers.py`, where the dynamic lateral acceleration cap and curvature clipping already live. Extend `clip_curvature()` to report whether the default `3.0 m/s^2` cap would have saturated the requested curvature, then feed that signal into `update_lateral_accel_limit()` from `controlsd.py` on the next control loop. This intentionally avoids a new subsystem, UI, params, or torque-controller changes.

**Tech Stack:** Python, pytest, sunnypilot/openpilot controls helpers, `uv run` for tests.

---

## File Structure

- Modify `selfdrive/controls/lib/drive_helpers.py`: add a saturation input to `update_lateral_accel_limit()`, add a helper for default-cap saturation detection, and return that signal from `clip_curvature()`.
- Modify `selfdrive/controls/controlsd.py`: track the previous cycle's default-cap saturation signal and pass it into `update_lateral_accel_limit()`.
- Modify `selfdrive/controls/tests/test_drive_helpers.py`: add deterministic tests for burst activation, non-activation, decay, resets, and manual-gas precedence.
- Use `docs/superpowers/specs/2026-05-01-lateral-accel-curve-entry-burst-design.md` as the approved design reference.

---

### Task 1: Add Burst Tests To Drive Helpers

**Files:**
- Modify: `selfdrive/controls/tests/test_drive_helpers.py`

- [ ] **Step 1: Add failing tests for burst state transitions**

Add these tests after `test_lateral_accel_limit_blocks_driver_gas_override_during_driver_intervention()`:

```python

def test_lateral_accel_limit_enters_burst_after_default_cap_saturation():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=True,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL)


def test_lateral_accel_limit_does_not_burst_without_default_cap_saturation():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=False,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)


def test_lateral_accel_limit_refreshes_burst_while_default_cap_saturation_continues():
  limit = update_lateral_accel_limit(
    4.0,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=True,
    dt=0.5,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL)
```

- [ ] **Step 2: Run the focused test file and verify failure**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_drive_helpers.py -q
```

Expected: FAIL with `TypeError: update_lateral_accel_limit() got an unexpected keyword argument 'default_lateral_accel_limited'`.

- [ ] **Step 3: Add failing tests for burst reset, decay, and manual-gas precedence**

Add these tests after the tests from Step 1:

```python

def test_lateral_accel_limit_decays_after_burst_saturation_ends():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
    manual_gas_override=False,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=False,
    dt=0.5,
  )

  expected = MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - (
    (MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - MAX_LATERAL_ACCEL_NO_ROLL) / 1.25
  ) * 0.5
  assert limit == pytest.approx(expected)


@pytest.mark.parametrize(
  "lat_active,brake_pressed,steering_pressed",
  [
    (False, False, False),
    (True, True, False),
    (True, False, True),
  ],
)
def test_lateral_accel_limit_resets_burst_for_inactive_or_driver_intervention(lat_active, brake_pressed, steering_pressed):
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
    manual_gas_override=False,
    lat_active=lat_active,
    brake_pressed=brake_pressed,
    steering_pressed=steering_pressed,
    default_lateral_accel_limited=True,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)


def test_lateral_accel_limit_driver_gas_wins_over_burst_state():
  limit = update_lateral_accel_limit(
    MAX_LATERAL_ACCEL_NO_ROLL,
    manual_gas_override=True,
    lat_active=True,
    brake_pressed=False,
    steering_pressed=False,
    default_lateral_accel_limited=False,
  )

  assert limit == pytest.approx(MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL)
```

- [ ] **Step 4: Run the focused test file and verify failure remains signature-related**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_drive_helpers.py -q
```

Expected: FAIL with `TypeError: update_lateral_accel_limit() got an unexpected keyword argument 'default_lateral_accel_limited'`.

---

### Task 2: Implement Burst State In Drive Helpers

**Files:**
- Modify: `selfdrive/controls/lib/drive_helpers.py`
- Test: `selfdrive/controls/tests/test_drive_helpers.py`

- [ ] **Step 1: Extend `update_lateral_accel_limit()` signature and logic**

Replace the current `update_lateral_accel_limit()` function in `selfdrive/controls/lib/drive_helpers.py` with:

```python
def update_lateral_accel_limit(current_limit, manual_gas_override, lat_active, brake_pressed, steering_pressed,
                               default_lateral_accel_limited=False, dt=DT_CTRL):
  if not lat_active or brake_pressed or steering_pressed or not np.isfinite(current_limit):
    return MAX_LATERAL_ACCEL_NO_ROLL
  if manual_gas_override or default_lateral_accel_limited:
    return MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL

  decay_rate = (MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL - MAX_LATERAL_ACCEL_NO_ROLL) / LATERAL_ACCEL_DRIVER_GAS_DECAY_SECONDS
  return float(np.clip(current_limit - decay_rate * max(dt, 0.0),
                       MAX_LATERAL_ACCEL_NO_ROLL,
                       MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL))
```

- [ ] **Step 2: Run focused tests and verify they pass**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_drive_helpers.py -q
```

Expected: PASS with all tests in `test_drive_helpers.py` passing.

---

### Task 3: Report Default-Cap Saturation From `clip_curvature()`

**Files:**
- Modify: `selfdrive/controls/lib/drive_helpers.py`
- Modify: `selfdrive/controls/tests/test_drive_helpers.py`

- [ ] **Step 1: Add failing tests for the third return value**

Add these tests after `test_clip_curvature_driver_gas_still_respects_max_curvature()`:

```python

def test_clip_curvature_reports_default_lateral_accel_limit_saturation():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited, default_lateral_accel_limited = clip_curvature(
    v_ego,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL / v_ego**2)
  assert limited
  assert default_lateral_accel_limited


def test_clip_curvature_reports_no_default_saturation_when_request_fits_default_cap():
  v_ego = 10.0
  requested_curvature = 2.0 / v_ego**2

  clipped_curvature, limited, default_lateral_accel_limited = clip_curvature(
    v_ego,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(requested_curvature)
  assert not limited
  assert not default_lateral_accel_limited


def test_clip_curvature_reports_default_saturation_even_when_burst_cap_allows_request():
  v_ego = 10.0
  requested_curvature = 4.0 / v_ego**2

  clipped_curvature, limited, default_lateral_accel_limited = clip_curvature(
    v_ego,
    requested_curvature,
    requested_curvature,
    0.0,
    MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL,
  )

  assert clipped_curvature == pytest.approx(requested_curvature)
  assert not limited
  assert default_lateral_accel_limited
```

- [ ] **Step 2: Update existing tuple unpacking in the same test file**

In `selfdrive/controls/tests/test_drive_helpers.py`, change existing two-value `clip_curvature()` unpacking to three values:

```python
clipped_curvature, limited, _ = clip_curvature(v_ego, requested_curvature, requested_curvature, 0.0)
```

For the multi-line calls, change the assignment line from:

```python
clipped_curvature, limited = clip_curvature(
```

to:

```python
clipped_curvature, limited, _ = clip_curvature(
```

- [ ] **Step 3: Run focused tests and verify failure**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_drive_helpers.py -q
```

Expected: FAIL with `ValueError: not enough values to unpack (expected 3, got 2)`.

- [ ] **Step 4: Add a helper for default lateral-accel saturation detection**

In `selfdrive/controls/lib/drive_helpers.py`, add this helper between `update_lateral_accel_limit()` and `clip_curvature()`:

```python
def is_default_lateral_accel_limited(v_ego, curvature, roll):
  v_ego = max(v_ego, MIN_SPEED)
  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  max_lat_accel = MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  min_lat_accel = -MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  lat_accel = curvature * v_ego ** 2
  return lat_accel < min_lat_accel or lat_accel > max_lat_accel
```

- [ ] **Step 5: Return default-cap saturation from `clip_curvature()`**

In `selfdrive/controls/lib/drive_helpers.py`, update `clip_curvature()` so it computes the default-cap saturation after jerk limiting and before the active cap clamp:

```python
def clip_curvature(v_ego, prev_curvature, new_curvature, roll, lateral_accel_limit=MAX_LATERAL_ACCEL_NO_ROLL) -> tuple[float, bool, bool]:
  # This function respects ISO lateral jerk and acceleration limits + a max curvature
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)
  default_lateral_accel_limited = is_default_lateral_accel_limited(v_ego, new_curvature, roll)

  if not np.isfinite(lateral_accel_limit):
    lateral_accel_limit = MAX_LATERAL_ACCEL_NO_ROLL
  lateral_accel_limit = float(np.clip(lateral_accel_limit, 0.0, MAX_LATERAL_ACCEL_DRIVER_GAS_NO_ROLL))

  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  max_lat_accel = lateral_accel_limit + roll_compensation
  min_lat_accel = -lateral_accel_limit + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)

  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv, bool(default_lateral_accel_limited)
```

- [ ] **Step 6: Run focused tests and verify they pass**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_drive_helpers.py -q
```

Expected: PASS with all tests in `test_drive_helpers.py` passing.

---

### Task 4: Wire Saturation Latch Through `controlsd.py`

**Files:**
- Modify: `selfdrive/controls/controlsd.py`
- Test: `selfdrive/controls/tests/test_drive_helpers.py`

- [ ] **Step 1: Find `Controls.__init__()` and add saturation state**

In `selfdrive/controls/controlsd.py`, find the other `self.*` state initialization near `self.lateral_accel_limit_no_roll`. Add:

```python
self.default_lateral_accel_limited = False
```

If `self.lateral_accel_limit_no_roll` is initialized as:

```python
self.lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
```

the final block should include both lines:

```python
self.lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
self.default_lateral_accel_limited = False
```

- [ ] **Step 2: Pass previous-cycle saturation into the limit update**

In `selfdrive/controls/controlsd.py`, update the call at lines around 196-202 from:

```python
self.lateral_accel_limit_no_roll = update_lateral_accel_limit(
  self.lateral_accel_limit_no_roll,
  manual_gas_lateral_accel_override,
  CC.latActive,
  CS.brakePressed,
  CS.steeringPressed,
)
```

to:

```python
self.lateral_accel_limit_no_roll = update_lateral_accel_limit(
  self.lateral_accel_limit_no_roll,
  manual_gas_lateral_accel_override,
  CC.latActive,
  CS.brakePressed,
  CS.steeringPressed,
  self.default_lateral_accel_limited,
)
```

- [ ] **Step 3: Store the current saturation returned by `clip_curvature()`**

In `selfdrive/controls/controlsd.py`, update the call at lines around 203-209 from:

```python
self.desired_curvature, curvature_limited = clip_curvature(
  CS.vEgo,
  self.desired_curvature,
  new_desired_curvature,
  lp.roll,
  self.lateral_accel_limit_no_roll,
)
```

to:

```python
self.desired_curvature, curvature_limited, self.default_lateral_accel_limited = clip_curvature(
  CS.vEgo,
  self.desired_curvature,
  new_desired_curvature,
  lp.roll,
  self.lateral_accel_limit_no_roll,
)
```

- [ ] **Step 4: Run focused helper tests**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_drive_helpers.py -q
```

Expected: PASS with all tests in `test_drive_helpers.py` passing.

- [ ] **Step 5: Search for remaining two-value `clip_curvature()` unpacking**

Run:

```bash
rg "[^_,[:alnum:]]clip_curvature\(" selfdrive sunnypilot
```

Expected: only the updated `controlsd.py`, `drive_helpers.py`, and test calls appear; no remaining two-value assignment from `clip_curvature()`.

---

### Task 5: Final Verification

**Files:**
- Verify: `selfdrive/controls/lib/drive_helpers.py`
- Verify: `selfdrive/controls/controlsd.py`
- Verify: `selfdrive/controls/tests/test_drive_helpers.py`
- Verify: `docs/superpowers/specs/2026-05-01-lateral-accel-curve-entry-burst-design.md`

- [ ] **Step 1: Run focused drive helper tests**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_drive_helpers.py -q
```

Expected: PASS with all tests in `test_drive_helpers.py` passing.

- [ ] **Step 2: Run adjacent lateral helper tests that previously passed**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_model_path_processor.py selfdrive/controls/tests/test_lane_change_path_shaper.py -q
```

Expected: PASS with all tests in both files passing.

- [ ] **Step 3: Inspect the diff for scope control**

Run:

```bash
git diff -- selfdrive/controls/lib/drive_helpers.py selfdrive/controls/controlsd.py selfdrive/controls/tests/test_drive_helpers.py docs/superpowers/specs/2026-05-01-lateral-accel-curve-entry-burst-design.md docs/superpowers/plans/2026-05-01-lateral-accel-curve-entry-burst.md
```

Expected: diff only touches the planned files and does not change torque-controller shaping, `MAX_CURVATURE`, UI, params, or submodules.

- [ ] **Step 4: Commit only if explicitly requested by the user**

Do not commit by default. If the user explicitly asks for a commit, stage only the planned files:

```bash
git add selfdrive/controls/lib/drive_helpers.py selfdrive/controls/controlsd.py selfdrive/controls/tests/test_drive_helpers.py docs/superpowers/specs/2026-05-01-lateral-accel-curve-entry-burst-design.md docs/superpowers/plans/2026-05-01-lateral-accel-curve-entry-burst.md
git commit -m "controls: add lateral accel burst on saturation"
```

Expected: commit succeeds without staging unrelated files.

---

## Self-Review Notes

- Spec coverage: default cap preservation, manual-gas `5.0 m/s^2`, saturation-only burst activation, any-speed operation, reset conditions, no UI/param/torque changes, and focused testing are all covered by Tasks 1-5.
- Placeholder scan: no unresolved placeholders are intentionally left in the plan.
- Type consistency: `clip_curvature()` changes from returning `tuple[float, bool]` to `tuple[float, bool, bool]`; all known callers are updated in the plan.
