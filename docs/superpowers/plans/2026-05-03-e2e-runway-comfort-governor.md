# E2E Runway Comfort Governor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runway-aware comfort governor so long-runway no-lead E2E slowdowns prefer coast/light braking while preserving urgent model, stop-approach, and MPC lead braking.

**Architecture:** Implement one helper in `selfdrive/controls/lib/longitudinal_planner.py` that filters only the raw no-lead E2E acceleration candidate before the existing `min(output_a_target_e2e, output_a_target_mpc)` arbitration. The helper uses model endpoint distance, coast acceleration, a required-decel estimate, and a mild negative ramp limit. Tests cover helper behavior directly, including radar-lead and engage-bootstrap passthrough.

**Tech Stack:** Python, cereal-style `SimpleNamespace` test doubles, `uv run --extra testing pytest`, `ruff`.

---

## File Structure

- Modify `selfdrive/controls/lib/longitudinal_planner.py`
  - Add E2E runway comfort constants near existing E2E stop-approach constants.
  - Add helper `get_e2e_runway_comfort_accel(...)` near `get_e2e_stop_approach_accel(...)`.
  - Call the helper in `LongitudinalPlanner.update()` immediately after `output_a_target_e2e` is read and before E2E/MPC arbitration.
- Modify `selfdrive/controls/tests/test_longitudinal_planner.py`
  - Import the new helper.
  - Add focused helper-level tests for long runway, short runway, stop/driver/reset/radar/bootstrap gates, and ramp limiting.

## Task 1: Add Failing Helper Tests

**Files:**
- Modify: `selfdrive/controls/tests/test_longitudinal_planner.py:3-98`
- Test: `selfdrive/controls/tests/test_longitudinal_planner.py`

- [ ] **Step 1: Import the new helper in the test file**

Add `get_e2e_runway_comfort_accel` to the import list:

```python
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  E2E_STOP_APPROACH_DECEL_MAX,
  get_e2e_runway_comfort_accel,
  get_e2e_stop_approach_accel,
  has_model_stop_context,
  has_valid_radar_lead,
  should_run_engage_stop_bootstrap,
)
```

- [ ] **Step 2: Add long-runway coast-first test**

Append this test after the existing E2E stop-approach tests:

```python
def test_e2e_runway_comfort_caps_long_runway_raw_model_braking():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert -0.45 <= accel <= -0.25
```

- [ ] **Step 3: Add short-runway passthrough test**

Append this test:

```python
def test_e2e_runway_comfort_allows_short_runway_model_braking():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=55.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -1.2
```

- [ ] **Step 4: Add stop and override gate tests**

Append these tests:

```python
def test_e2e_runway_comfort_leaves_model_stop_untouched():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-1.2,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-1.2, should_stop=True, endpoint_x=145.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert accel == -1.2


def test_e2e_runway_comfort_leaves_driver_override_untouched():
  model_msg = make_model_msg(desired_accel=-1.2, should_stop=False, endpoint_x=145.0)

  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, gas_pressed=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, brake_pressed=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, reset_state=True) == -1.2
  assert get_e2e_runway_comfort_accel(17.2, -1.2, -0.25, model_msg, True, -0.2, force_slow_decel=True) == -1.2
```

- [ ] **Step 5: Add ramp limit test**

Append this test:

```python
def test_e2e_runway_comfort_limits_negative_ramp():
  accel = get_e2e_runway_comfort_accel(
    v_ego=17.2,
    raw_e2e_accel=-0.8,
    coast_accel=-0.25,
    model_msg=make_model_msg(desired_accel=-0.8, should_stop=False, endpoint_x=90.0),
    e2e_active=True,
    prev_output_a_target=-0.2,
  )

  assert -0.5 < accel < -0.2
```

- [ ] **Step 6: Run tests to verify they fail**

Run:

```bash
uv run --extra testing pytest selfdrive/controls/tests/test_longitudinal_planner.py -q
```

Expected: FAIL with an import error for `get_e2e_runway_comfort_accel`.

## Task 2: Implement E2E Runway Comfort Helper

**Files:**
- Modify: `selfdrive/controls/lib/longitudinal_planner.py:101-184`
- Test: `selfdrive/controls/tests/test_longitudinal_planner.py`

- [ ] **Step 1: Add constants near existing E2E constants**

Add these constants after `E2E_STOP_APPROACH_DECEL_MAX = 1.2`:

```python
E2E_RUNWAY_COMFORT_MIN_V_EGO = 3.0
E2E_RUNWAY_COMFORT_MIN_ENDPOINT = 1.0
E2E_RUNWAY_COMFORT_COAST_MARGIN = 0.12
E2E_RUNWAY_COMFORT_LIGHT_DECEL = 0.45
E2E_RUNWAY_COMFORT_DECEL_BLEND_BP = [0.35, 0.95]
E2E_RUNWAY_COMFORT_RUNWAY_BLEND_BP = [1.2, 2.0]
E2E_RUNWAY_COMFORT_NEGATIVE_RAMP_RATE = 0.35
```

- [ ] **Step 2: Add helper implementation**

Add this function after `get_e2e_stop_approach_accel(...)`:

```python
def get_e2e_runway_comfort_accel(v_ego, raw_e2e_accel, coast_accel, model_msg, e2e_active, prev_output_a_target,
                                 reset_state=False, force_slow_decel=False, brake_pressed=False, gas_pressed=False,
                                 dt=DT_MDL):
  blocked = not e2e_active or reset_state or force_slow_decel or brake_pressed or gas_pressed
  blocked = blocked or model_msg.action.shouldStop or v_ego < E2E_RUNWAY_COMFORT_MIN_V_EGO
  blocked = blocked or raw_e2e_accel >= coast_accel
  blocked = blocked or len(model_msg.position.x) == 0
  if blocked:
    return raw_e2e_accel

  endpoint_x = float(model_msg.position.x[-1])
  if not np.isfinite(endpoint_x) or endpoint_x <= E2E_RUNWAY_COMFORT_MIN_ENDPOINT:
    return raw_e2e_accel

  required_decel = v_ego**2 / (2.0 * endpoint_x)
  runway_ratio = endpoint_x / max(v_ego**2 / (2.0 * max(required_decel, E2E_RUNWAY_COMFORT_MIN_ENDPOINT)), E2E_RUNWAY_COMFORT_MIN_ENDPOINT)
  # `required_decel` already captures runway urgency; keep this value finite and easy to test.
  urgency_blend = float(np.interp(required_decel, E2E_RUNWAY_COMFORT_DECEL_BLEND_BP, [0.0, 1.0]))
  runway_blend = float(np.interp(runway_ratio, E2E_RUNWAY_COMFORT_RUNWAY_BLEND_BP, [1.0, 0.0]))
  blend = max(urgency_blend, runway_blend)

  light_decel_cap = min(coast_accel - E2E_RUNWAY_COMFORT_COAST_MARGIN, -E2E_RUNWAY_COMFORT_LIGHT_DECEL)
  comfort_cap = (1.0 - blend) * light_decel_cap + blend * raw_e2e_accel
  governed_accel = max(raw_e2e_accel, comfort_cap)

  max_negative_step = E2E_RUNWAY_COMFORT_NEGATIVE_RAMP_RATE * max(dt, 0.0)
  if np.isfinite(prev_output_a_target):
    governed_accel = max(governed_accel, prev_output_a_target - max_negative_step)
  return governed_accel
```

- [ ] **Step 3: Run helper tests**

Run:

```bash
uv run --extra testing pytest selfdrive/controls/tests/test_longitudinal_planner.py -q
```

Expected: The new tests may reveal that the runway-ratio expression is ineffective. If any expected long-runway/short-runway case fails, proceed to Task 3 to refine the helper calculation.

## Task 3: Refine Runway Blend and Wire Helper Into Planner

**Files:**
- Modify: `selfdrive/controls/lib/longitudinal_planner.py:158-184,616-624`
- Test: `selfdrive/controls/tests/test_longitudinal_planner.py`

- [ ] **Step 1: Replace the runway ratio with explicit expected-distance comparison**

In `get_e2e_runway_comfort_accel`, replace the `required_decel` through `blend` calculation with this version:

```python
  expected_distance = float(np.interp(v_ego * CV.MS_TO_KPH, E2E_STOP_APPROACH_EXPECTED_DIST_BP, E2E_STOP_APPROACH_EXPECTED_DIST_V))
  max_decel_distance = v_ego**2 / (2.0 * E2E_STOP_APPROACH_DECEL_MAX * (1.0 - E2E_STOP_APPROACH_SHORTAGE_BP[0]))
  expected_distance = max(expected_distance, max_decel_distance)
  if expected_distance <= 0.0:
    return raw_e2e_accel

  required_decel = v_ego**2 / (2.0 * endpoint_x)
  runway_ratio = endpoint_x / expected_distance
  urgency_blend = float(np.interp(required_decel, E2E_RUNWAY_COMFORT_DECEL_BLEND_BP, [0.0, 1.0]))
  runway_blend = float(np.interp(runway_ratio, E2E_RUNWAY_COMFORT_RUNWAY_BLEND_BP, [1.0, 0.0]))
  blend = max(urgency_blend, runway_blend)
```

- [ ] **Step 2: Wire helper into `LongitudinalPlanner.update()`**

Replace:

```python
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop
```

with:

```python
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop
    e2e_active = self.is_e2e(sm)
    output_a_target_e2e = get_e2e_runway_comfort_accel(
      v_ego, output_a_target_e2e, accel_coast, sm['modelV2'], e2e_active, prev_output_a_target,
      reset_state=reset_state,
      force_slow_decel=force_slow_decel,
      brake_pressed=sm['carState'].brakePressed,
      gas_pressed=sm['carState'].gasPressed,
      dt=self.dt,
    )
```

Then remove the later duplicate assignment immediately before the `if e2e_active:` block:

```python
    e2e_active = self.is_e2e(sm)
```

The final block should look like:

```python
    if e2e_active:
      output_a_target_e2e = apply_lead_loss_e2e_guard_accel(
        output_a_target_e2e, output_should_stop_e2e, self.lead_loss_e2e_guard_timer, has_radar_lead
      )
```

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run --extra testing pytest selfdrive/controls/tests/test_longitudinal_planner.py -q
```

Expected: PASS.

## Task 4: Add Regression Test for Existing Stop-Approach Helper Interaction

**Files:**
- Modify: `selfdrive/controls/tests/test_longitudinal_planner.py`
- Test: `selfdrive/controls/tests/test_longitudinal_planner.py`

- [ ] **Step 1: Add test proving shortage helper can still be stronger**

Append this test:

```python
def test_e2e_runway_comfort_does_not_block_stop_approach_shortage_braking():
  model_msg = make_model_msg(desired_accel=-0.4, should_stop=False, endpoint_x=45.0)
  governed = get_e2e_runway_comfort_accel(12.0, -0.4, -0.25, model_msg, True, -0.2)
  shortage_accel = get_e2e_stop_approach_accel(12.0, model_msg, make_radar_state(), True)

  assert shortage_accel < governed
  assert shortage_accel < -0.5
```

- [ ] **Step 2: Run tests**

Run:

```bash
uv run --extra testing pytest selfdrive/controls/tests/test_longitudinal_planner.py -q
```

Expected: PASS.

## Task 5: Lint and Verify

**Files:**
- Verify: `selfdrive/controls/lib/longitudinal_planner.py`
- Verify: `selfdrive/controls/tests/test_longitudinal_planner.py`

- [ ] **Step 1: Run ruff**

Run:

```bash
uv run --extra testing ruff check selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_longitudinal_planner.py
```

Expected: PASS. If it fails on line length or unused constants, fix only the reported issue.

- [ ] **Step 2: Run focused pytest again**

Run:

```bash
uv run --extra testing pytest selfdrive/controls/tests/test_longitudinal_planner.py -q
```

Expected: PASS.

- [ ] **Step 3: Check git diff**

Run:

```bash
git diff -- selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_longitudinal_planner.py docs/superpowers/specs/2026-05-03-e2e-runway-comfort-governor-design.md docs/superpowers/plans/2026-05-03-e2e-runway-comfort-governor.md
```

Expected: Diff contains only the spec, plan, helper, wiring, and tests described above.

---

## Self-Review Notes

- Spec coverage: Tasks cover the governor helper, planner data flow, safety gates, helper tests, and verification commands from the spec.
- Placeholder scan: No `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency: The helper signature used in tests matches the implementation and planner wiring.
