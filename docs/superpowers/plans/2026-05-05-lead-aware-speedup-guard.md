# Lead-Aware Speed-Up Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent SLA and cruise target generation from accelerating into a close, closing radar lead.

**Architecture:** Add a focused helper in the SP longitudinal planner that detects close closing lead conditions and applies it before target selection. The helper only caps positive speed-up seeds for cruise and speed-limit assist; SCC/map/OSM lower advisory targets continue to compete normally.

**Tech Stack:** Python, pytest, cereal capnp enum types, sunnypilot longitudinal planner target-selection helpers.

---

## File Structure

- Modify: `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
  - Add constants for close/closing lead guard thresholds.
  - Add `should_block_lead_speedup(...)` and `apply_lead_speedup_guard(...)` helper functions near the existing speed-limit target helpers.
  - Call the guard inside `LongitudinalPlannerSP.update_targets(...)` before selecting the lowest target.
- Modify: `sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py`
  - Extend fake `SubMaster` objects with `radarState` and lead fields.
  - Add unit tests for SLA and cruise speed-up blocking plus non-blocking cases.

---

### Task 1: Add Failing Unit Tests

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py`

- [ ] **Step 1: Add fake lead helpers**

Add this helper code after `FakeSubMaster`:

```python
def make_sm(v_cruise_cluster=20.0, lead_status=False, d_rel=100.0, v_rel=0.0,
            gas_pressed=False, brake_pressed=False):
  return FakeSubMaster({
    'carState': SimpleNamespace(
      vCruiseCluster=v_cruise_cluster,
      gasPressed=gas_pressed,
      brakePressed=brake_pressed,
    ),
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
    'radarState': SimpleNamespace(leadOne=SimpleNamespace(status=lead_status, dRel=d_rel, vRel=v_rel)),
  })
```

- [ ] **Step 2: Update existing test setup to use `make_sm` where target tests call `update_targets`**

For tests that currently create this repeated `FakeSubMaster`:

```python
sm = FakeSubMaster({
  'carState': SimpleNamespace(vCruiseCluster=20.0),
  'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
})
```

replace it with:

```python
sm = make_sm(v_cruise_cluster=20.0)
```

For tests using `vCruiseCluster=67.0`, use:

```python
sm = make_sm(v_cruise_cluster=67.0)
```

- [ ] **Step 3: Add failing test for SLA speed-up blocked by close closing lead**

Add this test after `test_speed_limit_auto_uses_assist_source_without_acceleration_seed`:

```python
def test_speed_limit_auto_speedup_blocked_by_close_closing_lead():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = FakeSpeedLimitAssist()
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=20.0, lead_status=True, d_rel=20.0, v_rel=-1.4)

  v_ego = 15.0
  v_cruise = 20.0 * CV.KPH_TO_MS
  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=v_ego, a_ego=0.2, v_cruise=v_cruise)

  assert planner.source == LongitudinalPlanSource.speedLimitAssist
  assert v_target == pytest.approx(v_ego)
  assert a_target <= 0.0
```

- [ ] **Step 4: Add failing test for cruise speed-up blocked by close closing lead**

Add this test after the SLA blocking test:

```python
def test_cruise_speedup_blocked_by_close_closing_lead():
  planner = LongitudinalPlannerSP.__new__(LongitudinalPlannerSP)
  planner.scc = FakeSmartCruiseControl()
  planner.resolver = FakeResolver()
  planner.sla = SequenceSpeedLimitAssist(active_sequence=[False], target_sequence=[255.0])
  planner.osm_traffic_control_prior = FakeOsmTrafficControlPrior()
  planner.events_sp = SimpleNamespace()
  planner.source = LongitudinalPlanSource.cruise
  planner._speed_limit_handoff_active = False
  planner._speed_limit_active_prev = False

  sm = make_sm(v_cruise_cluster=67.0, lead_status=True, d_rel=20.0, v_rel=-1.4)

  v_target, a_target = LongitudinalPlannerSP.update_targets(planner, sm, v_ego=15.0, a_ego=0.2, v_cruise=18.61)

  assert planner.source == LongitudinalPlanSource.cruise
  assert v_target == pytest.approx(15.0)
  assert a_target <= 0.0
```

- [ ] **Step 5: Add non-blocking tests**

Add these tests after the cruise blocking test:

```python
def test_lead_speedup_guard_allows_far_lead():
  assert not longitudinal_planner.should_block_lead_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=60.0,
    v_rel=-1.4,
    gas_pressed=False,
    brake_pressed=False,
  )


def test_lead_speedup_guard_allows_opening_lead():
  assert not longitudinal_planner.should_block_lead_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=20.0,
    v_rel=0.2,
    gas_pressed=False,
    brake_pressed=False,
  )


def test_lead_speedup_guard_allows_driver_gas_override():
  assert not longitudinal_planner.should_block_lead_speedup(
    v_ego=15.0,
    lead_status=True,
    d_rel=20.0,
    v_rel=-1.4,
    gas_pressed=True,
    brake_pressed=False,
  )
```

- [ ] **Step 6: Run tests and verify failure**

Run:

```bash
uv run --extra testing --extra tools python -m pytest sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py -q
```

Expected: tests fail because `should_block_lead_speedup` is not defined and target outputs are not yet guarded.

---

### Task 2: Implement Lead-Aware Guard

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`

- [ ] **Step 1: Add guard constants and helpers**

Add below the existing speed-limit constants:

```python
LEAD_SPEEDUP_GUARD_TIME_GAP = 2.2  # s, match the observed uncomfortable closing window.
LEAD_SPEEDUP_GUARD_MIN_DISTANCE = 25.0  # m, low-speed floor for close-lead gating.
LEAD_SPEEDUP_GUARD_CLOSING_V_REL = -0.2  # m/s, ignore noise around matched speed.
LEAD_SPEEDUP_GUARD_A_TARGET_MAX = 0.0  # m/s^2, coast instead of accelerating into the lead.


def should_block_lead_speedup(v_ego: float, lead_status: bool, d_rel: float, v_rel: float,
                              gas_pressed: bool, brake_pressed: bool) -> bool:
  if not lead_status or gas_pressed or brake_pressed:
    return False
  if v_rel > LEAD_SPEEDUP_GUARD_CLOSING_V_REL:
    return False

  close_distance = max(LEAD_SPEEDUP_GUARD_MIN_DISTANCE, v_ego * LEAD_SPEEDUP_GUARD_TIME_GAP)
  return d_rel < close_distance


def apply_lead_speedup_guard(active: bool, v_ego: float, target: tuple[float, float]) -> tuple[float, float]:
  if not active:
    return target

  v_target, a_target = target
  return min(v_target, v_ego), min(a_target, LEAD_SPEEDUP_GUARD_A_TARGET_MAX)
```

- [ ] **Step 2: Call guard in `update_targets`**

After `cruise_target` is computed, add:

```python
    lead_one = sm['radarState'].leadOne
    lead_speedup_guard_active = should_block_lead_speedup(
      v_ego,
      bool(lead_one.status),
      float(lead_one.dRel),
      float(lead_one.vRel),
      bool(CS.gasPressed),
      bool(CS.brakePressed),
    )
    speed_limit_assist_target = apply_lead_speedup_guard(lead_speedup_guard_active, v_ego, speed_limit_assist_target)
    cruise_target = apply_lead_speedup_guard(lead_speedup_guard_active, v_ego, cruise_target)
```

Keep this before `select_lowest_longitudinal_target(...)` so the guarded targets are used by source selection.

- [ ] **Step 3: Run focused tests**

Run:

```bash
uv run --extra testing --extra tools python -m pytest sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py -q
```

Expected: all tests pass.

---

### Task 3: Verify Route-Relevant Regressions

**Files:**
- No code changes.

- [ ] **Step 1: Run speed-limit and longitudinal target tests**

Run:

```bash
uv run --extra testing --extra tools python -m pytest sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_assist.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run follow-distance smoke coverage**

Run:

```bash
uv run --extra testing --extra tools python -m pytest selfdrive/controls/tests/test_following_distance.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Review diff**

Run:

```bash
git diff -- sunnypilot/selfdrive/controls/lib/longitudinal_planner.py sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py
```

Expected: diff is limited to lead-aware target guarding and tests.

---

### Task 4: Commit and Propagation Handoff

**Files:**
- Modified files from Tasks 1-2.
- Added docs from this spec/plan session.

- [ ] **Step 1: Check status**

Run:

```bash
git status -sb
```

Expected: only the planned files are modified or added.

- [ ] **Step 2: Commit retained-branch fix**

Run:

```bash
git add docs/superpowers/specs/2026-05-05-lead-aware-speedup-guard-design.md docs/superpowers/plans/2026-05-05-lead-aware-speedup-guard.md sunnypilot/selfdrive/controls/lib/longitudinal_planner.py sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py
git commit -m "speed-limit: guard speed-up near closing leads"
```

Expected: commit succeeds on `feat/speed-limit-auto-cruise`.

- [ ] **Step 3: Propagate downstream from custom worktree**

Run from `/home/jjolano/Developer/projects/sunnypilot`:

```bash
scripts/propagate-retained.sh --from feat/speed-limit-auto-cruise
```

Expected: downstream retained branches merge cleanly and push by default.

- [ ] **Step 4: Rebuild and deploy custom after propagation**

Run from `/home/jjolano/Developer/projects/sunnypilot`:

```bash
scripts/rebuild-custom.sh
scripts/deploy.sh --host comma@10.0.1.205
```

Expected: custom rebuild succeeds and device deploy completes. If `10.0.1.205` is unreachable, test `comma@10.0.50.205` before retrying deploy.
