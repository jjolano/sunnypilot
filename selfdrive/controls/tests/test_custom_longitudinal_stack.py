from dataclasses import replace

import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v1 import (
  CUSTOM_V1_FLOOR,
  CustomLongitudinalStackV1,
  CustomV1Candidate,
  select_custom_v1_candidate,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.registry import make_custom_longitudinal_stack
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V1


def make_output(a_target=-0.2, debug=None):
  return LongitudinalStackOutput(
    a_target=a_target,
    should_stop=False,
    has_lead=True,
    source="cruise",
    allow_throttle=True,
    allow_brake=True,
    speeds=tuple(10.0 for _ in range(CONTROL_N)),
    accels=tuple(a_target for _ in range(CONTROL_N)),
    jerks=tuple(0.0 for _ in range(CONTROL_N)),
    fcw=False,
    debug={} if debug is None else debug,
  )


def test_custom_v1_preserves_sunnypilot_output_and_marks_debug_boundary():
  original_debug = {"adapter": "sunnypilot-current"}
  sunnypilot_output = make_output(debug=original_debug)

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output)

  assert custom_output.a_target == sunnypilot_output.a_target
  assert custom_output.should_stop == sunnypilot_output.should_stop
  assert custom_output.has_lead == sunnypilot_output.has_lead
  assert custom_output.source == sunnypilot_output.source
  assert custom_output.allow_throttle == sunnypilot_output.allow_throttle
  assert custom_output.allow_brake == sunnypilot_output.allow_brake
  assert custom_output.speeds == sunnypilot_output.speeds
  assert custom_output.accels == sunnypilot_output.accels
  assert custom_output.jerks == sunnypilot_output.jerks
  assert custom_output.fcw == sunnypilot_output.fcw
  assert custom_output.debug == {
    "adapter": "sunnypilot-current",
    "custom_stack": CUSTOM_V1,
    "custom_v1_candidate": "sunnypilot-current",
    "custom_v1_mode": "candidate_arbitration",
  }
  assert original_debug == {"adapter": "sunnypilot-current"}


def test_custom_v1_candidate_arbiter_selects_most_restrictive_accel():
  cruise_candidate = CustomV1Candidate("cruise", make_output(a_target=0.1))
  lead_candidate = CustomV1Candidate("lead", make_output(a_target=-0.4))

  selected = select_custom_v1_candidate((cruise_candidate, lead_candidate))

  assert selected is lead_candidate


def test_custom_v1_can_actuate_more_restrictive_internal_candidate():
  sunnypilot_output = make_output(a_target=0.1, debug={"adapter": "sunnypilot-current"})
  stop_candidate = CustomV1Candidate("model_stop", make_output(a_target=-0.6))

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output, candidates=(stop_candidate,))

  assert custom_output.a_target == -0.6
  assert custom_output.debug["adapter"] == "sunnypilot-current"
  assert custom_output.debug["custom_v1_candidate"] == "model_stop"
  assert custom_output.debug["custom_v1_mode"] == "candidate_arbitration"


def test_custom_v1_can_actuate_relaxing_floor_candidate():
  sunnypilot_output = make_output(a_target=-1.0, debug={"adapter": "sunnypilot-current"})
  coast_candidate = CustomV1Candidate("cruise_coast", make_output(a_target=-0.3), selection=CUSTOM_V1_FLOOR)

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output, candidates=(coast_candidate,))

  assert custom_output.a_target == -0.3
  assert custom_output.debug["custom_v1_candidate"] == "cruise_coast"


def test_custom_v1_restrictive_candidate_wins_over_conflicting_floor():
  sunnypilot_output = make_output(a_target=-0.8, debug={"adapter": "sunnypilot-current"})
  coast_candidate = CustomV1Candidate("cruise_coast", make_output(a_target=-0.3), selection=CUSTOM_V1_FLOOR)
  stop_candidate = CustomV1Candidate("model_stop", make_output(a_target=-0.6))

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output, candidates=(coast_candidate, stop_candidate))

  assert custom_output.a_target == -0.6
  assert custom_output.debug["custom_v1_candidate"] == "model_stop"


def test_custom_v1_scoped_cap_does_not_cap_unrelated_floor():
  sunnypilot_output = make_output(a_target=0.0, debug={"adapter": "sunnypilot-current"})
  creep_cap = CustomV1Candidate("creep_cap", make_output(a_target=0.55), group="creep_to_stop_gap")
  launch_floor = CustomV1Candidate("launch", make_output(a_target=0.7), selection=CUSTOM_V1_FLOOR, group="launch")

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output, candidates=(creep_cap, launch_floor))

  assert custom_output.a_target == 0.7
  assert custom_output.debug["custom_v1_candidate"] == "launch"


def test_custom_v1_stop_intent_does_not_relax_stronger_baseline_braking():
  sunnypilot_output = make_output(a_target=-3.0, debug={"adapter": "sunnypilot-current"})
  stop_candidate = CustomV1Candidate("stop_guard", make_output(a_target=-2.0, debug={"reason": "stop_guard"}))
  stop_candidate = CustomV1Candidate(stop_candidate.name, replace(stop_candidate.output, should_stop=True))

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output, candidates=(stop_candidate,))

  assert custom_output.a_target == -3.0
  assert custom_output.should_stop
  assert custom_output.debug["custom_v1_candidate"] == "stop_guard"


def test_custom_v1_post_cap_rate_floor_limits_negative_jerk():
  sunnypilot_output = make_output(a_target=0.8, debug={"adapter": "sunnypilot-current"})
  creep_cap = CustomV1Candidate("creep_to_stop_gap", make_output(a_target=0.0), group="creep_to_stop_gap")
  rate_floor = CustomV1Candidate(
    "low_speed_pullaway_accel_step_floor",
    make_output(a_target=0.4),
    selection=CUSTOM_V1_FLOOR,
    group="low_speed_pullaway_accel_step",
  )

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output, candidates=(creep_cap, rate_floor))

  assert custom_output.a_target == 0.4
  assert custom_output.debug["custom_v1_candidate"] == "low_speed_pullaway_accel_step_floor"


def test_custom_v1_candidate_arbiter_rejects_empty_candidate_set():
  with pytest.raises(ValueError, match="no_custom_v1_candidates"):
    select_custom_v1_candidate(())


def test_custom_stack_factory_builds_custom_v1():
  stack = make_custom_longitudinal_stack(CUSTOM_V1)

  assert isinstance(stack, CustomLongitudinalStackV1)
  assert stack.stack_name == CUSTOM_V1


def test_custom_stack_factory_rejects_unknown_stack():
  with pytest.raises(ValueError, match="unsupported_custom_stack:custom-9.9"):
    make_custom_longitudinal_stack("custom-9.9")
