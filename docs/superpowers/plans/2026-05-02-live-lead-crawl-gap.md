# Live Lead Crawl Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the profiled low-speed lead-crawl gap behavior to live longitudinal follow-gap control.

**Architecture:** Keep the change in `selfdrive/controls/lib/longitudinal_planner.py`, where existing creep-to-stop-gap helpers already gate and shape low-speed stopped-lead movement. The helper will wait until the confirmed lead has opened at least `stop_target + 2.0 m`, crawl toward a `stop_target + 1.0 m` follow zone, taper through the final meter, and disengage once speed/gap leaves the low-speed crawl window so normal MPC lead following resumes.

**Tech Stack:** Python, NumPy interpolation, existing openpilot longitudinal planner helpers, `pytest` via `uv run`.

---

## File Structure

- Modify `selfdrive/controls/lib/longitudinal_planner.py`: adjust crawl-gap constants and helper thresholds only.
- Modify `selfdrive/controls/tests/test_following_distance.py`: update existing stopped-lead creep tests and add explicit coverage for the new start/follow/soft-stop/normal-transition behavior.
- No cereal schema, UI, submodule, or custom-only workflow files change.

## Task 1: Add Live Crawl Profile Tests

**Files:**
- Modify: `selfdrive/controls/tests/test_following_distance.py`

- [ ] **Step 1: Import the new named crawl-profile constants**

In `selfdrive/controls/tests/test_following_distance.py`, update the `longitudinal_planner` import block so it includes these constants:

```python
  CREEP_TO_STOP_GAP_FOLLOW_EXCESS,
  CREEP_TO_STOP_GAP_START_EXCESS,
```

The block should include them near the existing creep constants:

```python
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CREEP_TO_STOP_GAP_ACCEL_MAX,
  CREEP_TO_STOP_GAP_ACCEL_MIN,
  CREEP_TO_STOP_GAP_FOLLOW_EXCESS,
  CREEP_TO_STOP_GAP_HOLD_EXCESS,
  CREEP_TO_STOP_GAP_MAX_EXCESS,
  CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX,
  CREEP_TO_STOP_GAP_PREDICT_MIN_GAP_OPENING,
  CREEP_TO_STOP_GAP_START_EXCESS,
  CREEP_TO_STOP_GAP_MODEL_LEAD_CAMERA_OFFSET,
  CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_X_STD,
  CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_Y_STD,
  CREEP_TO_STOP_GAP_MODEL_LEAD_MAX_V_STD,
  STOPPED_LEAD_GAP_FILL_ACCEL_MAX,
  STOPPED_LEAD_GAP_FILL_MAX_EXCESS,
  STOPPED_LEAD_GAP_FILL_MIN_EXCESS,
  get_stopped_lead_gap_fill_accel,
  get_creep_to_stop_gap_accel,
  get_model_lead_pullaway,
  get_predicted_lead_pullaway,
  has_predicted_lead_pullaway,
  should_arm_stopped_lead_gap_fill,
  should_hold_creep_to_stop_gap,
  should_release_creep_stop_hold,
)
```

- [ ] **Step 2: Replace the old arm/taper test with the profiled behavior**

Replace `test_creep_to_stop_gap_release_arms_and_tapers_to_target_gap` with:

```python
def test_creep_to_stop_gap_release_waits_for_profile_start_and_tapers_to_target_gap():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)

  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS - 0.1, 0.0, 1.0, False
  )
  assert not active
  assert accel == pytest.approx(0.0)

  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.0, 1.0, False
  )
  assert active
  assert 0.0 < accel <= CREEP_TO_STOP_GAP_ACCEL_MAX

  active, follow_accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_FOLLOW_EXCESS, 0.0, 1.0, active
  )
  assert active
  assert 0.0 < follow_accel < accel

  active, soft_stop_accel = get_creep_to_stop_gap_accel(
    0.2, stop_target + 0.3, 0.0, 1.0, active
  )
  assert active
  assert CREEP_TO_STOP_GAP_ACCEL_MIN < soft_stop_accel < 0.0

  active, accel = get_creep_to_stop_gap_accel(0.0, stop_target, 0.0, 1.0, active)
  assert not active
  assert accel == pytest.approx(0.0)
```

- [ ] **Step 3: Replace the adaptive stopped target test**

Replace `test_creep_to_stop_gap_uses_adaptive_stopped_target` with:

```python
def test_creep_to_stop_gap_uses_adaptive_stopped_target_profile_start():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.0, 0.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.0, 1.0, False
  )

  assert stop_target < STOP_DISTANCE
  assert active
  assert 0.0 < accel <= CREEP_TO_STOP_GAP_ACCEL_MAX
```

- [ ] **Step 4: Update pullaway and prediction tests to start at the profiled threshold**

In these tests, replace the old `STOP_DISTANCE + 0.6`, `STOP_DISTANCE + 0.7`, `STOP_DISTANCE + 1.0`, and `STOP_DISTANCE + 0.35` crawl-start distances with `STOP_DISTANCE + CREEP_TO_STOP_GAP_START_EXCESS` unless the test is intentionally checking a below-threshold blocker:

```python
def test_creep_to_stop_gap_uses_stronger_accel_for_confirmed_pullaway():
  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_START_EXCESS, 0.8, 1.0, False)

  assert active
  assert 0.40 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX


def test_creep_to_stop_gap_confirmed_pullaway_can_match_no_lead_launch_cap():
  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_START_EXCESS, 1.2, 1.0, False)

  assert active
  assert CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX == pytest.approx(0.55)
  assert accel == pytest.approx(CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX)


def test_creep_to_stop_gap_uses_firmer_floor_for_initial_pullaway():
  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_START_EXCESS, 0.25, 1.0, False)

  assert active
  assert 0.30 <= accel <= STOPPED_LEAD_GAP_FILL_ACCEL_MAX


def test_creep_to_stop_gap_smooths_confirmed_pullaway_step():
  active, accel = get_creep_to_stop_gap_accel(0.0, STOP_DISTANCE + CREEP_TO_STOP_GAP_START_EXCESS, 0.8, 1.0, False)

  assert active
  assert 0.30 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX
```

For `test_creep_to_stop_gap_uses_predicted_pullaway_before_speed_threshold`, use a stopped target and a start distance:

```python
def test_creep_to_stop_gap_uses_predicted_pullaway_before_speed_threshold():
  stop_target = get_lead_stop_presentation_distance(0.0, 0.05, 1.0, 1.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 0.05, 1.0, False, a_lead=1.0, a_lead_tau=0.0
  )

  assert active
  assert 0.40 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX
```

For `test_creep_to_stop_gap_uses_model_lead_pullaway_prediction`, use a distance that requires model-predicted opening to reach the `+2 m` profile start:

```python
def test_creep_to_stop_gap_uses_model_lead_pullaway_prediction():
  d_rel = STOP_DISTANCE + CREEP_TO_STOP_GAP_START_EXCESS - 0.5
  model_v_lead, model_gap_opening = get_model_lead_pullaway(make_model_msg_lead(d_rel), make_radar_lead(d_rel), 0.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, d_rel, 0.0, 1.0, False,
    model_predicted_v_lead=model_v_lead,
    model_predicted_gap_opening=model_gap_opening,
  )

  assert has_predicted_lead_pullaway(d_rel - STOP_DISTANCE, model_v_lead, model_gap_opening)
  assert active
  assert 0.40 <= accel <= CREEP_TO_STOP_GAP_PULLAWAY_ACCEL_MAX
```

- [ ] **Step 5: Add explicit normal-following transition coverage**

Add this test near the other `get_creep_to_stop_gap_accel` tests:

```python
def test_creep_to_stop_gap_transitions_to_normal_following_after_pullaway_window():
  stop_target = get_lead_stop_presentation_distance(0.0, 1.2, 0.6, 1.0)
  active, accel = get_creep_to_stop_gap_accel(
    0.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS, 1.2, 1.0, False, a_lead=0.6
  )
  assert active
  assert accel > 0.0

  active, accel = get_creep_to_stop_gap_accel(
    1.0, stop_target + CREEP_TO_STOP_GAP_START_EXCESS + 0.5, 1.8, 1.0, active, a_lead=0.6
  )
  assert not active
  assert accel == pytest.approx(0.0)

  active, accel = get_creep_to_stop_gap_accel(
    0.2, stop_target + CREEP_TO_STOP_GAP_MAX_EXCESS + 0.1, 1.8, 1.0, True, a_lead=0.6
  )
  assert not active
  assert accel == pytest.approx(0.0)
```

- [ ] **Step 6: Run the focused tests and verify RED**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_following_distance.py \
  -k 'creep_to_stop_gap or stopped_lead_gap_fill' -q
```

Expected before production changes: FAIL because `CREEP_TO_STOP_GAP_FOLLOW_EXCESS` and `CREEP_TO_STOP_GAP_START_EXCESS` do not exist yet, or because the below-start crawl assertion still activates under the old `+0.5 m` threshold.

## Task 2: Implement Minimal Live Crawl Profile Constants

**Files:**
- Modify: `selfdrive/controls/lib/longitudinal_planner.py`

- [ ] **Step 1: Add named crawl-profile excess constants**

Replace the existing stopped-gap creep constants at the top of `longitudinal_planner.py` with this minimal set of new thresholds while preserving unrelated constants:

```python
CREEP_TO_STOP_GAP_START_EXCESS = 2.0
CREEP_TO_STOP_GAP_FOLLOW_EXCESS = 1.0
CREEP_TO_STOP_GAP_ARM_EXCESS = CREEP_TO_STOP_GAP_START_EXCESS
CREEP_TO_STOP_GAP_STOP_EXCESS = 0.05
CREEP_TO_STOP_GAP_MAX_V_EGO_ARM = 0.3
CREEP_TO_STOP_GAP_MAX_V_EGO = 1.0
# Treat pullaway creep as a near stopped-gap behavior; farther leads return to normal MPC handling.
CREEP_TO_STOP_GAP_MAX_EXCESS = 4.0
CREEP_TO_STOP_GAP_MIN_LEAD_SPEED = -0.3
CREEP_TO_STOP_GAP_MIN_MODEL_PROB = 0.5
CREEP_TO_STOP_GAP_SPEED_MAX = 0.75
CREEP_TO_STOP_GAP_SPEED_BP = [
  CREEP_TO_STOP_GAP_STOP_EXCESS,
  CREEP_TO_STOP_GAP_FOLLOW_EXCESS,
  CREEP_TO_STOP_GAP_START_EXCESS,
  CREEP_TO_STOP_GAP_MAX_EXCESS,
]
CREEP_TO_STOP_GAP_SPEED_V = [0.0, 0.18, 0.30, CREEP_TO_STOP_GAP_SPEED_MAX]
CREEP_TO_STOP_GAP_ACCEL_GAIN = 0.8
CREEP_TO_STOP_GAP_ACCEL_MIN = -0.25
CREEP_TO_STOP_GAP_ACCEL_MAX = 0.18
CREEP_TO_STOP_GAP_HOLD_EXCESS = 0.3
CREEP_TO_STOP_GAP_REHOLD_EXCESS = 0.2
CREEP_TO_STOP_GAP_HOLD_RELEASE_EXCESS = CREEP_TO_STOP_GAP_START_EXCESS
CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_SPEED = 0.05
CREEP_TO_STOP_GAP_HOLD_RELEASE_MIN_LEAD_ACCEL = 0.15
CREEP_TO_STOP_GAP_PULLAWAY_MIN_LEAD_SPEED = 0.25
CREEP_TO_STOP_GAP_PULLAWAY_ARM_EXCESS = CREEP_TO_STOP_GAP_START_EXCESS
```

Leave all subsequent pullaway prediction, model lead, stopped lead gap fill, planner update, and recovery constants unchanged.

- [ ] **Step 2: Run the focused tests and verify GREEN for crawl helpers**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_following_distance.py \
  -k 'creep_to_stop_gap or stopped_lead_gap_fill' -q
```

Expected: PASS for the focused crawl and stopped-lead gap-fill tests.

- [ ] **Step 3: Run the whole follow-distance test file**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_following_distance.py -q
```

Expected: PASS.

## Task 3: Verify Integration Boundaries

**Files:**
- Read-only verification for `selfdrive/controls/lib/longitudinal_planner.py`
- Read-only verification for `selfdrive/controls/tests/test_following_distance.py`

- [ ] **Step 1: Verify the decision-layer tests still pass**

Run:

```bash
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: PASS. This confirms the opt-in decision layer still composes with legacy planner behavior.

- [ ] **Step 2: Run lint on modified Python files**

Run:

```bash
uv run ruff check selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_following_distance.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Inspect local diff**

Run:

```bash
git diff -- selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_following_distance.py docs/superpowers/plans/2026-05-02-live-lead-crawl-gap.md
```

Expected: diff is limited to crawl-profile constants/tests and this plan.

## Task 4: Commit Follow-Gap Work

**Files:**
- Stage: `selfdrive/controls/lib/longitudinal_planner.py`
- Stage: `selfdrive/controls/tests/test_following_distance.py`
- Stage with force if ignored: `docs/superpowers/plans/2026-05-02-live-lead-crawl-gap.md`

- [ ] **Step 1: Check repository state**

Run:

```bash
git status --short --branch
git diff -- selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_following_distance.py docs/superpowers/plans/2026-05-02-live-lead-crawl-gap.md
git log --oneline -5
```

Expected: only the intended follow-gap files and plan are modified/untracked.

- [ ] **Step 2: Stage intended files**

Run:

```bash
git add selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_following_distance.py
git add -f docs/superpowers/plans/2026-05-02-live-lead-crawl-gap.md
```

Expected: files staged.

- [ ] **Step 3: Commit**

Run:

```bash
git commit -m "controlsd: profile low-speed lead crawl gap"
```

Expected: commit succeeds without hook failures.

## Task 5: Propagate, Rebuild, Verify, Deploy

**Files:**
- No direct manual edits expected.

- [ ] **Step 1: Return to the custom admin worktree**

Run commands from `/home/jjolano/Developer/projects/sunnypilot`.

- [ ] **Step 2: Propagate retained branches from follow-gap**

Run:

```bash
uv run scripts/propagate-retained.sh --from feat/longitudinal-follow-gap --no-push
```

Expected: propagation completes locally. If a downstream retained branch is checked out in another worktree, stop and report the path instead of forcing it closed.

- [ ] **Step 3: Rebuild custom**

Run:

```bash
uv run scripts/rebuild-custom.sh
```

Expected: `custom` rebuild completes and restores custom-only files.

- [ ] **Step 4: Verify rebuilt custom**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_following_distance.py -q
uv run pytest --confcutdir=selfdrive/controls/tests selfdrive/controls/tests/test_longitudinal_decision.py -q
uv run pytest tools/drive_lab/tests -q
uv run pytest sunnypilot/sunnylink/tests/test_params_sync.py -q
```

Expected: all listed tests pass.

- [ ] **Step 5: Deploy custom**

Run:

```bash
uv run scripts/deploy.sh
```

Expected: deploy script completes and reboots the device by default.

- [ ] **Step 6: Run deploy health check**

Run the documented deploy health check commands from `AGENTS.md` if deploy completes and SSH is expected to be available.

Expected: deployed commit matches local `custom`, manager/core processes are running, no recent obvious traceback/import crash loop, and retained-feature imports succeed.

## Self-Review

- Spec coverage: The plan covers live `+2 m` start, `+1 m` follow zone, final-meter soft-stop, blocker preservation, normal-following transition after pullaway, focused tests, integration tests, propagation, rebuild, and deploy.
- Placeholder scan: No TBD/TODO placeholders remain.
- Type consistency: New constants are imported from and defined in `longitudinal_planner.py`; tests call existing helper signatures without introducing new APIs.
