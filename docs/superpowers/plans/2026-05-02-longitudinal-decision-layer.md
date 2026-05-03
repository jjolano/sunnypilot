# Longitudinal Decision Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature-flagged longitudinal decision layer that normalizes planner feature outputs into candidates, arbitrates them through explicit policy, exposes rich Python-side telemetry, and adds an opt-in UI toggle defaulting off.

**Architecture:** Add a focused `longitudinal_decision.py` module for candidate types, arbitration, fallback resolution, and candidate construction helpers. Keep `LongControl` unchanged, preserve the current planner as fallback authority, and integrate the new layer at the end of `LongitudinalPlanner.update()` behind `LongitudinalDecisionLayer`.

**Tech Stack:** Python 3, pytest via `uv run`, openpilot params, sunnypilot Python UI settings, existing `cereal` enums, existing longitudinal planner tests.

---

## File Structure

- Create `selfdrive/controls/lib/longitudinal_decision.py`
  - Owns `CandidateRole`, `DecisionSource`, `LongitudinalCandidate`, `LongitudinalDecision`, `LongitudinalArbiter`, fallback resolution, and small candidate construction helpers.
- Create `selfdrive/controls/tests/test_longitudinal_decision.py`
  - Unit coverage for candidate validation, arbitration policy, fallback behavior, telemetry, and core candidate construction.
- Modify `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
  - Adds a pure helper for sunnypilot candidate construction and stores `self.decision_candidates_sp` after existing SP target updates.
- Modify `sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py`
  - Tests SP candidate construction without changing existing lowest-target behavior tests.
- Modify `selfdrive/controls/lib/longitudinal_planner.py`
  - Instantiates params and arbiter, collects core plus SP candidates, resolves final output behind the toggle, and stores telemetry on `self.longitudinal_decision`.
- Modify `common/params_keys.h`
  - Adds persistent backup bool param `LongitudinalDecisionLayer` with default `0`.
- Modify `selfdrive/ui/sunnypilot/layouts/settings/cruise.py`
  - Adds the default-off opt-in UI toggle under Cruise settings.
- Modify `sunnypilot/sunnylink/params_metadata.json`
  - Adds metadata for `LongitudinalDecisionLayer` so sunnylink metadata stays in sync.
- Modify `sunnypilot/sunnylink/tests/test_params_sync.py`
  - Adds a known-param check for the new metadata entry.

## Task 1: Candidate Model And Validation

**Files:**
- Create: `selfdrive/controls/lib/longitudinal_decision.py`
- Create: `selfdrive/controls/tests/test_longitudinal_decision.py`

- [ ] **Step 1: Write failing candidate model tests**

Add this file:

```python
import math

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  CandidateRole,
  DecisionSource,
  LongitudinalCandidate,
)


def test_candidate_clamps_confidence_and_urgency():
  candidate = LongitudinalCandidate(
    source=DecisionSource.CRUISE,
    role=CandidateRole.DRIVER_INTENT,
    v_target=20.0,
    a_target=0.1,
    confidence=2.0,
    urgency=-1.0,
    active_reason="driver_set_speed",
  )

  assert candidate.confidence == 1.0
  assert candidate.urgency == 0.0


def test_candidate_rejects_non_finite_targets():
  candidate = LongitudinalCandidate(
    source=DecisionSource.CRUISE,
    role=CandidateRole.DRIVER_INTENT,
    v_target=math.inf,
    a_target=0.0,
    confidence=1.0,
    urgency=0.0,
    active_reason="bad_speed",
  )

  assert not candidate.valid
  assert "non_finite" in candidate.invalid_reason


def test_candidate_rejects_empty_reason():
  candidate = LongitudinalCandidate(
    source=DecisionSource.SPEED_LIMIT,
    role=CandidateRole.ADVISORY_CAP,
    v_target=15.0,
    a_target=-0.2,
    confidence=0.8,
    urgency=0.4,
    active_reason="",
  )

  assert not candidate.valid
  assert candidate.invalid_reason == "missing_active_reason"


def test_candidate_records_debug_without_affecting_validity():
  candidate = LongitudinalCandidate(
    source=DecisionSource.SCC_MAP,
    role=CandidateRole.ADVISORY_CAP,
    v_target=12.0,
    a_target=-0.3,
    confidence=0.9,
    urgency=0.5,
    active_reason="confident_map_curve",
    debug={"distance": 80.0},
  )

  assert candidate.valid
  assert candidate.debug == {"distance": 80.0}
```

- [ ] **Step 2: Run candidate tests and verify they fail**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: FAIL with `ModuleNotFoundError` or import errors for `longitudinal_decision` symbols.

- [ ] **Step 3: Implement candidate model**

Create `selfdrive/controls/lib/longitudinal_decision.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


class CandidateRole(Enum):
  DRIVER_INTENT = "driver_intent"
  PHYSICAL_HAZARD = "physical_hazard"
  ADVISORY_CAP = "advisory_cap"
  COMFORT_SHAPING = "comfort_shaping"
  FALLBACK = "fallback"


class DecisionSource(Enum):
  CRUISE = "cruise"
  LEAD_MPC = "lead_mpc"
  E2E_STOP = "e2e_stop"
  SPEED_LIMIT = "speed_limit"
  SCC_VISION = "scc_vision"
  SCC_MAP = "scc_map"
  OSM_TRAFFIC_CONTROL = "osm_traffic_control"
  CRUISE_COAST = "cruise_coast"
  STOP_LAUNCH = "stop_launch"
  LEGACY_FALLBACK = "legacy_fallback"


def _clamp01(value: float) -> float:
  if not math.isfinite(value):
    return 0.0
  return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class LongitudinalCandidate:
  source: DecisionSource
  role: CandidateRole
  v_target: float
  a_target: float
  confidence: float
  urgency: float
  active_reason: str
  should_stop: bool = False
  comfort_bounds: tuple[float | None, float | None] = (None, None)
  safety_bounds: tuple[float | None, float | None] = (None, None)
  debug: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    object.__setattr__(self, "v_target", float(self.v_target))
    object.__setattr__(self, "a_target", float(self.a_target))
    object.__setattr__(self, "confidence", _clamp01(float(self.confidence)))
    object.__setattr__(self, "urgency", _clamp01(float(self.urgency)))
    object.__setattr__(self, "active_reason", str(self.active_reason))

  @property
  def invalid_reason(self) -> str:
    if not math.isfinite(self.v_target) or not math.isfinite(self.a_target):
      return "non_finite_target"
    if self.v_target < 0.0:
      return "negative_v_target"
    if not self.active_reason:
      return "missing_active_reason"
    return ""

  @property
  def valid(self) -> bool:
    return self.invalid_reason == ""
```

- [ ] **Step 4: Run candidate tests and verify they pass**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: PASS with 4 tests.

- [ ] **Step 5: Commit candidate model**

Run:

```bash
git add selfdrive/controls/lib/longitudinal_decision.py selfdrive/controls/tests/test_longitudinal_decision.py
git commit -m "feat: add longitudinal decision candidates"
```

## Task 2: Arbiter Policy And Fallback Resolution

**Files:**
- Modify: `selfdrive/controls/lib/longitudinal_decision.py`
- Modify: `selfdrive/controls/tests/test_longitudinal_decision.py`

- [ ] **Step 1: Add failing arbiter and fallback tests**

Append these tests to `selfdrive/controls/tests/test_longitudinal_decision.py`:

```python
from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  LongitudinalArbiter,
  resolve_longitudinal_decision,
)


def make_candidate(source, role, v_target, a_target, confidence, urgency, reason, should_stop=False):
  return LongitudinalCandidate(
    source=source,
    role=role,
    v_target=v_target,
    a_target=a_target,
    confidence=confidence,
    urgency=urgency,
    active_reason=reason,
    should_stop=should_stop,
  )


def test_arbiter_defaults_to_driver_intent():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")

  decision = arbiter.decide([cruise])

  assert decision.enabled
  assert decision.winner == DecisionSource.CRUISE
  assert decision.a_target == 0.2
  assert decision.suppressed == []


def test_confirmed_physical_hazard_overrides_speed_limit_advisory():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.1, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 18.0, -0.2, 0.9, 0.3, "advisory_limit")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 20.0, -0.8, 0.8, 0.7, "confirmed_lead")

  decision = arbiter.decide([cruise, speed_limit, lead])

  assert decision.winner == DecisionSource.LEAD_MPC
  assert decision.a_target == -0.8
  assert (DecisionSource.SPEED_LIMIT, "physical_hazard_active") in decision.suppressed


def test_high_confidence_advisory_cap_shapes_driver_intent_without_hazard():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.2, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 20.0, -0.3, 0.9, 0.4, "advisory_limit")

  decision = arbiter.decide([cruise, speed_limit])

  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert decision.a_target == -0.3


def test_low_confidence_advisory_is_suppressed_for_driver_intent():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.2, 1.0, 0.1, "driver_set_speed")
  weak_curve = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 18.0, -0.4, 0.3, 0.6, "weak_curve")

  decision = arbiter.decide([cruise, weak_curve])

  assert decision.winner == DecisionSource.CRUISE
  assert (DecisionSource.SCC_VISION, "low_confidence") in decision.suppressed


def test_comfort_candidate_can_relax_accel_when_no_hazard_or_advisory():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, -0.8, 1.0, 0.1, "driver_set_speed")
  coast = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.COMFORT_SHAPING, 25.0, -0.2, 0.8, 0.2, "harmless_overspeed")

  decision = arbiter.decide([cruise, coast])

  assert decision.winner == DecisionSource.CRUISE_COAST
  assert decision.a_target == -0.2


def test_resolver_returns_legacy_fallback_when_toggle_disabled():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")

  decision = resolve_longitudinal_decision(
    enabled=False,
    candidates=[cruise],
    fallback_v_target=25.0,
    fallback_a_target=-0.4,
    fallback_should_stop=True,
    accel_limits=(-1.2, 1.0),
    arbiter=LongitudinalArbiter(),
  )

  assert not decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.a_target == -0.4
  assert decision.should_stop
  assert decision.fallback_reason == "feature_flag_disabled"


def test_resolver_falls_back_when_decision_exceeds_accel_limits():
  bad = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 15.0, -5.0, 1.0, 1.0, "bad_decel")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[bad],
    fallback_v_target=25.0,
    fallback_a_target=-0.4,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=LongitudinalArbiter(),
  )

  assert not decision.enabled
  assert decision.winner == DecisionSource.LEGACY_FALLBACK
  assert decision.fallback_reason == "decision_outside_accel_limits"
```

- [ ] **Step 2: Run arbiter tests and verify they fail**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: FAIL with import errors for `LongitudinalArbiter` and `resolve_longitudinal_decision`.

- [ ] **Step 3: Implement arbiter and fallback resolver**

Append this code to `selfdrive/controls/lib/longitudinal_decision.py`:

```python
PHYSICAL_CONFIDENCE_MIN = 0.55
ADVISORY_CONFIDENCE_MIN = 0.75
COMFORT_CONFIDENCE_MIN = 0.50


@dataclass(frozen=True)
class LongitudinalDecision:
  enabled: bool
  winner: DecisionSource
  v_target: float
  a_target: float
  should_stop: bool
  candidates: tuple[LongitudinalCandidate, ...] = ()
  suppressed: tuple[tuple[DecisionSource, str], ...] = ()
  fallback_reason: str = ""

  def inside_accel_limits(self, accel_limits: tuple[float, float]) -> bool:
    lo, hi = accel_limits
    return math.isfinite(self.a_target) and lo <= self.a_target <= hi


def _fallback_decision(v_target: float, a_target: float, should_stop: bool, reason: str) -> LongitudinalDecision:
  return LongitudinalDecision(
    enabled=False,
    winner=DecisionSource.LEGACY_FALLBACK,
    v_target=float(v_target),
    a_target=float(a_target),
    should_stop=bool(should_stop),
    fallback_reason=reason,
  )


class LongitudinalArbiter:
  def decide(self, candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...]) -> LongitudinalDecision:
    valid = [candidate for candidate in candidates if candidate.valid]
    suppressed: list[tuple[DecisionSource, str]] = [
      (candidate.source, candidate.invalid_reason) for candidate in candidates if not candidate.valid
    ]

    driver = next((candidate for candidate in valid if candidate.role == CandidateRole.DRIVER_INTENT), None)
    if driver is None:
      driver = LongitudinalCandidate(
        source=DecisionSource.LEGACY_FALLBACK,
        role=CandidateRole.FALLBACK,
        v_target=0.0,
        a_target=0.0,
        confidence=1.0,
        urgency=0.0,
        active_reason="missing_driver_intent",
      )

    physical = [
      candidate for candidate in valid
      if candidate.role == CandidateRole.PHYSICAL_HAZARD and candidate.confidence >= PHYSICAL_CONFIDENCE_MIN
    ]
    low_confidence = [
      candidate for candidate in valid
      if candidate.role in (CandidateRole.PHYSICAL_HAZARD, CandidateRole.ADVISORY_CAP)
      and candidate.confidence < (PHYSICAL_CONFIDENCE_MIN if candidate.role == CandidateRole.PHYSICAL_HAZARD else ADVISORY_CONFIDENCE_MIN)
    ]
    suppressed.extend((candidate.source, "low_confidence") for candidate in low_confidence)

    if physical:
      winner = min(physical, key=lambda candidate: (candidate.a_target, candidate.v_target))
      suppressed.extend(
        (candidate.source, "physical_hazard_active") for candidate in valid
        if candidate is not winner and candidate.role != CandidateRole.PHYSICAL_HAZARD
      )
    else:
      advisory = [
        candidate for candidate in valid
        if candidate.role == CandidateRole.ADVISORY_CAP
        and candidate.confidence >= ADVISORY_CONFIDENCE_MIN
        and candidate.v_target < driver.v_target
      ]
      if advisory:
        winner = min(advisory, key=lambda candidate: (candidate.v_target, candidate.a_target))
        suppressed.extend((candidate.source, "higher_advisory_target") for candidate in advisory if candidate is not winner)
      else:
        comfort = [
          candidate for candidate in valid
          if candidate.role == CandidateRole.COMFORT_SHAPING
          and candidate.confidence >= COMFORT_CONFIDENCE_MIN
          and candidate.a_target > driver.a_target
        ]
        winner = max(comfort, key=lambda candidate: candidate.a_target) if comfort else driver

    return LongitudinalDecision(
      enabled=True,
      winner=winner.source,
      v_target=winner.v_target,
      a_target=winner.a_target,
      should_stop=winner.should_stop,
      candidates=tuple(valid),
      suppressed=tuple(dict.fromkeys(suppressed)),
    )


def resolve_longitudinal_decision(enabled: bool, candidates: list[LongitudinalCandidate] | tuple[LongitudinalCandidate, ...],
                                  fallback_v_target: float, fallback_a_target: float, fallback_should_stop: bool,
                                  accel_limits: tuple[float, float], arbiter: LongitudinalArbiter) -> LongitudinalDecision:
  if not enabled:
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "feature_flag_disabled")

  try:
    decision = arbiter.decide(candidates)
  except Exception:
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "arbiter_exception")

  if not math.isfinite(decision.v_target) or not math.isfinite(decision.a_target):
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "decision_non_finite")
  if not decision.inside_accel_limits(accel_limits):
    return _fallback_decision(fallback_v_target, fallback_a_target, fallback_should_stop, "decision_outside_accel_limits")

  return decision
```

- [ ] **Step 4: Run arbiter tests and verify they pass**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: PASS with 11 tests.

- [ ] **Step 5: Commit arbiter and fallback logic**

Run:

```bash
git add selfdrive/controls/lib/longitudinal_decision.py selfdrive/controls/tests/test_longitudinal_decision.py
git commit -m "feat: arbitrate longitudinal candidates"
```

## Task 3: Sunnypilot Candidate Producers

**Files:**
- Modify: `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`
- Modify: `sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py`

- [ ] **Step 1: Add failing SP candidate producer tests**

Append these tests to `sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py`:

```python
from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource


def test_sp_candidate_builder_includes_cruise_and_active_advisories():
  candidates = longitudinal_planner.build_sp_longitudinal_candidates(
    speed_limit_active=True,
    cruise=(25.0, 0.1),
    scc_vision=(22.0, -0.2),
    scc_vision_active=True,
    scc_map=(24.0, -0.1),
    scc_map_active=False,
    speed_limit_assist=(20.0, -0.3),
    osm_traffic_control=(18.0, -0.4),
    osm_traffic_control_active=True,
  )

  assert [candidate.source for candidate in candidates] == [
    DecisionSource.CRUISE,
    DecisionSource.SPEED_LIMIT,
    DecisionSource.SCC_VISION,
    DecisionSource.OSM_TRAFFIC_CONTROL,
  ]
  assert all(candidate.valid for candidate in candidates)
  assert candidates[0].role == CandidateRole.DRIVER_INTENT
  assert all(candidate.role == CandidateRole.ADVISORY_CAP for candidate in candidates[1:])


def test_sp_candidate_builder_skips_inactive_advisories():
  candidates = longitudinal_planner.build_sp_longitudinal_candidates(
    speed_limit_active=False,
    cruise=(25.0, 0.1),
    scc_vision=(22.0, -0.2),
    scc_vision_active=False,
    scc_map=(24.0, -0.1),
    scc_map_active=False,
    speed_limit_assist=(20.0, -0.3),
    osm_traffic_control=(18.0, -0.4),
    osm_traffic_control_active=False,
  )

  assert [candidate.source for candidate in candidates] == [DecisionSource.CRUISE]
```

- [ ] **Step 2: Run SP candidate tests and verify they fail**

Run:

```bash
uv run pytest sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py -q
```

Expected: FAIL with `AttributeError` for `build_sp_longitudinal_candidates`.

- [ ] **Step 3: Implement SP candidate builder and store candidates**

In `sunnypilot/selfdrive/controls/lib/longitudinal_planner.py`, add this import below the existing imports:

```python
from openpilot.selfdrive.controls.lib.longitudinal_decision import CandidateRole, DecisionSource, LongitudinalCandidate
```

Add this helper after `select_lowest_longitudinal_target`:

```python
def build_sp_longitudinal_candidates(speed_limit_active, cruise, scc_vision, scc_vision_active, scc_map, scc_map_active,
                                     speed_limit_assist, osm_traffic_control, osm_traffic_control_active):
  cruise_v, cruise_a = cruise
  candidates = [LongitudinalCandidate(
    source=DecisionSource.CRUISE,
    role=CandidateRole.DRIVER_INTENT,
    v_target=cruise_v,
    a_target=cruise_a,
    confidence=1.0,
    urgency=0.1,
    active_reason="driver_cruise_target",
  )]

  if speed_limit_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.SPEED_LIMIT,
      role=CandidateRole.ADVISORY_CAP,
      v_target=speed_limit_assist[0],
      a_target=speed_limit_assist[1],
      confidence=0.85,
      urgency=0.35,
      active_reason="speed_limit_assist_active",
    ))
  if scc_vision_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.SCC_VISION,
      role=CandidateRole.ADVISORY_CAP,
      v_target=scc_vision[0],
      a_target=scc_vision[1],
      confidence=0.80,
      urgency=0.45,
      active_reason="confident_vision_curve",
    ))
  if scc_map_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.SCC_MAP,
      role=CandidateRole.ADVISORY_CAP,
      v_target=scc_map[0],
      a_target=scc_map[1],
      confidence=0.80,
      urgency=0.40,
      active_reason="confident_map_curve",
    ))
  if osm_traffic_control_active:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.OSM_TRAFFIC_CONTROL,
      role=CandidateRole.ADVISORY_CAP,
      v_target=osm_traffic_control[0],
      a_target=osm_traffic_control[1],
      confidence=0.75,
      urgency=0.55,
      active_reason="model_confirmed_map_caution",
    ))

  return candidates
```

In `LongitudinalPlannerSP.__init__`, add this field after `self.output_a_target = 0.`:

```python
    self.decision_candidates_sp = []
```

In `LongitudinalPlannerSP.update_targets`, add this block immediately before `self.source, self.output_v_target, self.output_a_target = select_lowest_longitudinal_target(...)`:

```python
    self.decision_candidates_sp = build_sp_longitudinal_candidates(
      self.sla.is_active,
      (v_cruise, a_ego),
      (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      self.scc.vision.is_active,
      (self.scc.map.output_v_target, self.scc.map.output_a_target),
      self.scc.map.is_active,
      (self.sla.output_v_target, self.sla.output_a_target),
      (self.osm_traffic_control_prior.output_v_target, self.osm_traffic_control_prior.output_a_target),
      self.osm_traffic_control_prior.active,
    )
```

- [ ] **Step 4: Run SP candidate tests and verify they pass**

Run:

```bash
uv run pytest sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py -q
```

Expected: PASS with existing target-selection tests and the two new candidate tests.

- [ ] **Step 5: Commit SP candidate producers**

Run:

```bash
git add sunnypilot/selfdrive/controls/lib/longitudinal_planner.py sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py
git commit -m "feat: collect sunnypilot longitudinal candidates"
```

## Task 4: Core Candidate Producers

**Files:**
- Modify: `selfdrive/controls/lib/longitudinal_decision.py`
- Modify: `selfdrive/controls/tests/test_longitudinal_decision.py`

- [ ] **Step 1: Add failing core candidate producer tests**

Append these tests to `selfdrive/controls/tests/test_longitudinal_decision.py`:

```python
from openpilot.selfdrive.controls.lib.longitudinal_decision import build_core_longitudinal_candidates


def test_core_candidate_builder_adds_confirmed_lead_candidate():
  candidates = build_core_longitudinal_candidates(
    has_lead=True,
    lead_confidence=0.82,
    v_cruise=27.0,
    a_cruise=0.1,
    output_a_target_mpc=-0.7,
    output_should_stop_mpc=False,
    e2e_active=False,
    output_a_target_e2e=0.0,
    output_should_stop_e2e=False,
    e2e_stop_approach_a_target=0.0,
    cruise_coast_applied=False,
    cruise_coast_a_target=0.0,
  )

  lead = next(candidate for candidate in candidates if candidate.source == DecisionSource.LEAD_MPC)
  assert lead.role == CandidateRole.PHYSICAL_HAZARD
  assert lead.confidence == pytest.approx(0.82)
  assert lead.a_target == -0.7


def test_core_candidate_builder_adds_e2e_stop_candidate_for_active_stop():
  candidates = build_core_longitudinal_candidates(
    has_lead=False,
    lead_confidence=0.0,
    v_cruise=27.0,
    a_cruise=0.1,
    output_a_target_mpc=0.0,
    output_should_stop_mpc=False,
    e2e_active=True,
    output_a_target_e2e=-1.0,
    output_should_stop_e2e=True,
    e2e_stop_approach_a_target=0.0,
    cruise_coast_applied=False,
    cruise_coast_a_target=0.0,
  )

  e2e = next(candidate for candidate in candidates if candidate.source == DecisionSource.E2E_STOP)
  assert e2e.role == CandidateRole.PHYSICAL_HAZARD
  assert e2e.should_stop
  assert e2e.confidence == pytest.approx(0.85)


def test_core_candidate_builder_adds_cruise_coast_comfort_candidate():
  candidates = build_core_longitudinal_candidates(
    has_lead=False,
    lead_confidence=0.0,
    v_cruise=25.0,
    a_cruise=-0.8,
    output_a_target_mpc=-0.8,
    output_should_stop_mpc=False,
    e2e_active=False,
    output_a_target_e2e=0.0,
    output_should_stop_e2e=False,
    e2e_stop_approach_a_target=0.0,
    cruise_coast_applied=True,
    cruise_coast_a_target=-0.2,
  )

  coast = next(candidate for candidate in candidates if candidate.source == DecisionSource.CRUISE_COAST)
  assert coast.role == CandidateRole.COMFORT_SHAPING
  assert coast.a_target == -0.2
```

- [ ] **Step 2: Run core candidate tests and verify they fail**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: FAIL with import error for `build_core_longitudinal_candidates`.

- [ ] **Step 3: Implement core candidate builder**

Append this function to `selfdrive/controls/lib/longitudinal_decision.py`:

```python
def build_core_longitudinal_candidates(has_lead: bool, lead_confidence: float, v_cruise: float, a_cruise: float,
                                       output_a_target_mpc: float, output_should_stop_mpc: bool,
                                       e2e_active: bool, output_a_target_e2e: float, output_should_stop_e2e: bool,
                                       e2e_stop_approach_a_target: float,
                                       cruise_coast_applied: bool, cruise_coast_a_target: float) -> list[LongitudinalCandidate]:
  candidates: list[LongitudinalCandidate] = []

  if has_lead:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.LEAD_MPC,
      role=CandidateRole.PHYSICAL_HAZARD,
      v_target=max(0.0, v_cruise),
      a_target=output_a_target_mpc,
      confidence=max(0.60, lead_confidence),
      urgency=0.70 if output_a_target_mpc < -0.3 or output_should_stop_mpc else 0.45,
      active_reason="confirmed_radar_lead",
      should_stop=output_should_stop_mpc,
    ))

  if e2e_active or output_should_stop_e2e or e2e_stop_approach_a_target < 0.0:
    e2e_accel = min(output_a_target_e2e, e2e_stop_approach_a_target) if e2e_stop_approach_a_target < 0.0 else output_a_target_e2e
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.E2E_STOP,
      role=CandidateRole.PHYSICAL_HAZARD,
      v_target=max(0.0, v_cruise),
      a_target=e2e_accel,
      confidence=0.85 if output_should_stop_e2e else 0.65,
      urgency=0.80 if output_should_stop_e2e else 0.55,
      active_reason="model_stop_or_slowdown",
      should_stop=output_should_stop_e2e,
    ))

  if cruise_coast_applied:
    candidates.append(LongitudinalCandidate(
      source=DecisionSource.CRUISE_COAST,
      role=CandidateRole.COMFORT_SHAPING,
      v_target=max(0.0, v_cruise),
      a_target=cruise_coast_a_target,
      confidence=0.80,
      urgency=0.20,
      active_reason="context_efficient_overspeed_coast",
      debug={"legacy_cruise_accel": float(a_cruise)},
    ))

  return candidates
```

- [ ] **Step 4: Run core candidate tests and verify they pass**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: PASS with all decision-layer tests.

- [ ] **Step 5: Commit core candidate producers**

Run:

```bash
git add selfdrive/controls/lib/longitudinal_decision.py selfdrive/controls/tests/test_longitudinal_decision.py
git commit -m "feat: build core longitudinal candidates"
```

## Task 5: Planner Integration Behind Feature Flag

**Files:**
- Modify: `selfdrive/controls/lib/longitudinal_planner.py`
- Modify: `selfdrive/controls/tests/test_longitudinal_decision.py`

- [ ] **Step 1: Add failing resolver telemetry test for planner-facing behavior**

Append this test to `selfdrive/controls/tests/test_longitudinal_decision.py`:

```python
def test_enabled_resolver_keeps_candidate_telemetry():
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.2, 1.0, 0.1, "driver_set_speed")
  speed_limit = make_candidate(DecisionSource.SPEED_LIMIT, CandidateRole.ADVISORY_CAP, 20.0, -0.3, 0.9, 0.4, "advisory_limit")

  decision = resolve_longitudinal_decision(
    enabled=True,
    candidates=[cruise, speed_limit],
    fallback_v_target=25.0,
    fallback_a_target=0.2,
    fallback_should_stop=False,
    accel_limits=(-1.2, 1.0),
    arbiter=LongitudinalArbiter(),
  )

  assert decision.enabled
  assert decision.winner == DecisionSource.SPEED_LIMIT
  assert [candidate.source for candidate in decision.candidates] == [DecisionSource.CRUISE, DecisionSource.SPEED_LIMIT]
  assert decision.fallback_reason == ""
```

- [ ] **Step 2: Run decision tests and verify the new telemetry assertion passes before planner wiring**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: PASS. This confirms the resolver already supports the planner-facing telemetry before touching the large planner file.

- [ ] **Step 3: Wire the decision layer into planner imports and init**

In `selfdrive/controls/lib/longitudinal_planner.py`, add this import with the other imports:

```python
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_decision import (
  LongitudinalArbiter,
  build_core_longitudinal_candidates,
  resolve_longitudinal_decision,
)
```

In `LongitudinalPlanner.__init__`, add these fields after `self.mpc = LongitudinalMpc(dt=dt)`:

```python
    self.params = Params()
    self.longitudinal_arbiter = LongitudinalArbiter()
    self.longitudinal_decision = None
    self.longitudinal_decision_candidates = []
```

- [ ] **Step 4: Capture cruise-coast candidate state in `update()`**

Replace the cruise-coast block near the end of `LongitudinalPlanner.update()` with this version:

```python
    cruise_coast_applied = False
    cruise_coast_a_target = output_a_target
    if should_apply_cruise_coast_overspeed(
      reset_state, force_slow_decel, e2e_active, has_lead, self.output_should_stop, self.source
    ):
      cruise_coast_a_target = apply_cruise_coast_overspeed(v_ego, v_cruise, cruise_coast_accel, output_a_target)
      cruise_coast_applied = cruise_coast_a_target != output_a_target
      output_a_target = cruise_coast_a_target
```

- [ ] **Step 5: Resolve final output before final accel clipping**

Immediately after the cruise-coast block from Step 4 and before `self.previous_lead_loss_status = bool(lead_one.status)`, add:

```python
    legacy_a_target = float(output_a_target)
    legacy_should_stop = bool(self.output_should_stop)
    lead_confidence = float(getattr(lead_one, "modelProb", 0.0)) if lead_one.status else 0.0
    self.longitudinal_decision_candidates = list(getattr(self, "decision_candidates_sp", [])) + build_core_longitudinal_candidates(
      has_lead=has_lead,
      lead_confidence=lead_confidence,
      v_cruise=v_cruise,
      a_cruise=self.a_desired,
      output_a_target_mpc=output_a_target_mpc,
      output_should_stop_mpc=output_should_stop_mpc,
      e2e_active=e2e_active,
      output_a_target_e2e=output_a_target_e2e,
      output_should_stop_e2e=output_should_stop_e2e,
      e2e_stop_approach_a_target=e2e_stop_approach_a_target,
      cruise_coast_applied=cruise_coast_applied,
      cruise_coast_a_target=cruise_coast_a_target,
    )
    self.longitudinal_decision = resolve_longitudinal_decision(
      enabled=self.params.get_bool("LongitudinalDecisionLayer"),
      candidates=self.longitudinal_decision_candidates,
      fallback_v_target=v_cruise,
      fallback_a_target=legacy_a_target,
      fallback_should_stop=legacy_should_stop,
      accel_limits=(accel_clip[0], accel_clip[1]),
      arbiter=self.longitudinal_arbiter,
    )
    if self.longitudinal_decision.enabled:
      output_a_target = self.longitudinal_decision.a_target
      self.output_should_stop = self.longitudinal_decision.should_stop
```

- [ ] **Step 6: Run focused tests after planner wiring**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py selfdrive/controls/tests/test_longitudinal_planner.py selfdrive/controls/tests/test_cruise_coast.py -q
```

Expected: PASS for decision-layer tests, longitudinal planner helper tests, and cruise-coast tests.

- [ ] **Step 7: Commit planner integration**

Run:

```bash
git add selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/tests/test_longitudinal_decision.py
git commit -m "feat: gate longitudinal decision layer in planner"
```

## Task 6: Param, Metadata, And UI Toggle

**Files:**
- Modify: `common/params_keys.h`
- Modify: `selfdrive/ui/sunnypilot/layouts/settings/cruise.py`
- Modify: `sunnypilot/sunnylink/params_metadata.json`
- Modify: `sunnypilot/sunnylink/tests/test_params_sync.py`

- [ ] **Step 1: Add failing metadata test**

In `sunnypilot/sunnylink/tests/test_params_sync.py`, add this assertion block inside `test_known_params_metadata()` after the `CustomAccLongPressIncrement` checks:

```python
  decision_layer = metadata.get("LongitudinalDecisionLayer")
  assert decision_layer is not None
  assert decision_layer["title"] == "Longitudinal Decision Layer"
  assert "experimental" in decision_layer["description"].lower()
```

- [ ] **Step 2: Run metadata test and verify it fails**

Run:

```bash
uv run pytest sunnypilot/sunnylink/tests/test_params_sync.py::test_known_params_metadata -q
```

Expected: FAIL because `LongitudinalDecisionLayer` metadata is missing.

- [ ] **Step 3: Add params key**

In `common/params_keys.h`, insert this line after the `LongitudinalPersonality` entry:

```cpp
    {"LongitudinalDecisionLayer", {PERSISTENT | BACKUP, BOOL, "0"}},
```

- [ ] **Step 4: Add metadata entry**

In `sunnypilot/sunnylink/params_metadata.json`, insert this JSON entry near `LongitudinalPersonality`:

```json
  "LongitudinalDecisionLayer": {
    "title": "Longitudinal Decision Layer",
    "description": "Enable the experimental longitudinal decision layer. When enabled, sunnypilot arbitrates cruise, lead, e2e, speed-limit, SCC, OSM, and coast candidates through a unified policy with legacy fallback."
  },
```

- [ ] **Step 5: Add Cruise settings toggle**

In `selfdrive/ui/sunnypilot/layouts/settings/cruise.py`, add this toggle after `self.dec_toggle`:

```python
    self.longitudinal_decision_layer_toggle = toggle_item_sp(
      title=tr("Longitudinal Decision Layer (Experimental)"),
      description=tr("Use the new unified longitudinal arbitration layer. This is opt-in and falls back to current planner behavior if disabled or invalid."),
      param="LongitudinalDecisionLayer")
```

Add the toggle to the `items` list immediately after `self.dec_toggle`:

```python
      self.longitudinal_decision_layer_toggle,
```

In `_update_state()`, inside the `if has_long or has_icbm:` branch after `self.dec_toggle.action_item.set_enabled(has_long)`, add:

```python
        self.longitudinal_decision_layer_toggle.action_item.set_enabled(has_long)
```

Inside the matching `else:` branch after `ui_state.params.remove("DynamicExperimentalControl")`, add:

```python
        ui_state.params.remove("LongitudinalDecisionLayer")
```

Inside the same `else:` branch after `self.dec_toggle.action_item.set_enabled(False)`, add:

```python
        self.longitudinal_decision_layer_toggle.action_item.set_enabled(False)
```

- [ ] **Step 6: Run metadata and import checks**

Run:

```bash
uv run pytest sunnypilot/sunnylink/tests/test_params_sync.py::test_known_params_metadata -q
uv run python -m compileall selfdrive/ui/sunnypilot/layouts/settings/cruise.py selfdrive/controls/lib/longitudinal_planner.py selfdrive/controls/lib/longitudinal_decision.py
```

Expected: metadata test PASS and compileall exits 0.

- [ ] **Step 7: Commit param and UI toggle**

Run:

```bash
git add common/params_keys.h selfdrive/ui/sunnypilot/layouts/settings/cruise.py sunnypilot/sunnylink/params_metadata.json sunnypilot/sunnylink/tests/test_params_sync.py
git commit -m "feat: add longitudinal decision layer toggle"
```

## Task 7: Overlapping Candidate Policy Tests

**Files:**
- Modify: `selfdrive/controls/tests/test_longitudinal_decision.py`
- Modify: `selfdrive/controls/lib/longitudinal_decision.py` if these tests reveal a policy mismatch.

- [ ] **Step 1: Add overlapping candidate tests**

Append these tests to `selfdrive/controls/tests/test_longitudinal_decision.py`:

```python
def test_driver_intent_wins_over_low_confidence_cut_in():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 27.0, 0.1, 1.0, 0.1, "driver_set_speed")
  cut_in = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 20.0, -0.6, 0.3, 0.8, "low_confidence_cut_in")

  decision = arbiter.decide([cruise, cut_in])

  assert decision.winner == DecisionSource.CRUISE
  assert (DecisionSource.LEAD_MPC, "low_confidence") in decision.suppressed


def test_osm_caution_does_not_override_confirmed_lead():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 25.0, 0.1, 1.0, 0.1, "driver_set_speed")
  osm = make_candidate(DecisionSource.OSM_TRAFFIC_CONTROL, CandidateRole.ADVISORY_CAP, 8.33, -0.4, 0.8, 0.5, "map_caution")
  lead = make_candidate(DecisionSource.LEAD_MPC, CandidateRole.PHYSICAL_HAZARD, 15.0, -0.7, 0.9, 0.8, "confirmed_lead")

  decision = arbiter.decide([cruise, osm, lead])

  assert decision.winner == DecisionSource.LEAD_MPC
  assert (DecisionSource.OSM_TRAFFIC_CONTROL, "physical_hazard_active") in decision.suppressed


def test_confident_curve_can_limit_overspeed_when_no_hazard():
  arbiter = LongitudinalArbiter()
  cruise = make_candidate(DecisionSource.CRUISE, CandidateRole.DRIVER_INTENT, 30.0, -0.1, 1.0, 0.1, "driver_set_speed")
  curve = make_candidate(DecisionSource.SCC_VISION, CandidateRole.ADVISORY_CAP, 18.0, -0.5, 0.85, 0.6, "confident_vision_curve")
  coast = make_candidate(DecisionSource.CRUISE_COAST, CandidateRole.COMFORT_SHAPING, 30.0, 0.0, 0.8, 0.2, "harmless_overspeed")

  decision = arbiter.decide([cruise, curve, coast])

  assert decision.winner == DecisionSource.SCC_VISION
  assert decision.a_target == -0.5
```

- [ ] **Step 2: Run overlapping policy tests**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py -q
```

Expected: PASS. If a test fails, adjust only `LongitudinalArbiter.decide()` to satisfy the stated behavior while keeping previous tests passing.

- [ ] **Step 3: Run the broader focused test set**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py selfdrive/controls/tests/test_longitudinal_planner.py selfdrive/controls/tests/test_cruise_coast.py sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py sunnypilot/sunnylink/tests/test_params_sync.py::test_known_params_metadata -q
```

Expected: PASS for all listed tests.

- [ ] **Step 4: Commit overlapping policy coverage**

Run:

```bash
git add selfdrive/controls/lib/longitudinal_decision.py selfdrive/controls/tests/test_longitudinal_decision.py
git commit -m "test: cover longitudinal decision interactions"
```

## Task 8: Final Verification And Branch Readiness

**Files:**
- No code changes expected.

- [ ] **Step 1: Run focused verification suite**

Run:

```bash
uv run pytest selfdrive/controls/tests/test_longitudinal_decision.py selfdrive/controls/tests/test_longitudinal_planner.py selfdrive/controls/tests/test_cruise_coast.py sunnypilot/selfdrive/controls/lib/speed_limit/tests/test_speed_limit_planner_targets.py sunnypilot/sunnylink/tests/test_params_sync.py::test_known_params_metadata -q
```

Expected: PASS for every listed test.

- [ ] **Step 2: Run compile verification**

Run:

```bash
uv run python -m compileall selfdrive/controls/lib/longitudinal_decision.py selfdrive/controls/lib/longitudinal_planner.py sunnypilot/selfdrive/controls/lib/longitudinal_planner.py selfdrive/ui/sunnypilot/layouts/settings/cruise.py
```

Expected: command exits 0.

- [ ] **Step 3: Inspect final diff**

Run:

```bash
git status --short --branch
git log --oneline -n 8
```

Expected: branch is `feat/longitudinal-decision-layer` with no uncommitted code changes after the final verification commit sequence.

- [ ] **Step 4: Record implementation notes for integration**

Add a short final message to the user that includes these exact facts and the observed verification evidence from Steps 1 and 2:

```text
Implemented on feat/longitudinal-decision-layer.
Feature flag/UI param: LongitudinalDecisionLayer, default off.
Runtime authority: legacy planner when toggle is off or decision output is invalid.
Telemetry location: LongitudinalPlanner.longitudinal_decision and LongitudinalPlanner.longitudinal_decision_candidates.
Verification evidence: report the exact Step 1 pytest command, exact Step 2 compileall command, and the observed pass/fail status for each command.
```

## Spec Coverage Self-Review

- Unified decision model: Tasks 1, 2, 3, 4, and 5.
- Behavior redesign policy: Tasks 2 and 7.
- Existing planner fallback: Tasks 2 and 5.
- LongControl isolation: Task 5 integrates before publish and does not edit `longcontrol.py`.
- UI toggle default off: Task 6.
- Rich Python-side telemetry: Tasks 2, 5, and 7.
- Tests before implementation: Every code task starts with a failing or confirming test step before implementation.
- Branch ownership: implementation stays on `feat/longitudinal-decision-layer`; `custom` metadata updates are not part of this code plan and should be handled when this branch is added to the retained workflow.
