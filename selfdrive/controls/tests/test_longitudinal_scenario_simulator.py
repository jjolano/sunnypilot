import math

import pytest

from openpilot.selfdrive.controls.tests.longitudinal_scenario_simulator import (
  default_longitudinal_scenarios,
  simulate_scenario,
)


def _scenario_result(name: str):
  scenarios = {scenario.name: scenario for scenario in default_longitudinal_scenarios()}
  return simulate_scenario(scenarios[name])


def test_default_longitudinal_scenarios_are_deterministic_and_finite():
  scenarios = default_longitudinal_scenarios()

  assert {scenario.name for scenario in scenarios} == {
    "lead_cut_in_out_flicker",
    "stopped_lead_pullaway",
    "no_lead_model_stop",
    "false_model_stop_in_acc",
    "speed_limit_drop",
    "map_curve_false_positive",
    "vision_curve_true_positive",
    "downhill_overspeed",
    "one_pedal_lift_off",
  }
  for scenario in scenarios:
    first = simulate_scenario(scenario)
    second = simulate_scenario(scenario)
    assert first == second
    assert len(first) == len(scenario.frames)
    for frame in first:
      assert frame.scenario == scenario.name
      assert frame.requested_mode in ("acc", "e2e", "scc")
      assert frame.resolved_implementation
      assert frame.actuation_type
      assert frame.source
      assert frame.stack_intent
      assert frame.stack_reason
      assert isinstance(frame.selected_candidates, tuple)
      assert isinstance(frame.rejected_candidates, tuple)
      assert math.isfinite(frame.a_target)


def test_lead_cut_in_out_flicker_never_authorizes_progress():
  clear, new_close, shadow = _scenario_result("lead_cut_in_out_flicker")

  assert clear.source == "cruise"
  assert new_close.primary_physical_lead_idx == 0
  assert shadow.shadow_active
  for frame in (new_close, shadow):
    assert not frame.lead_progress_allowed
    assert frame.a_target <= 0.0
    assert frame.source == "lead_mpc"
    assert frame.stack_intent == "lead_follow"


def test_stopped_lead_pullaway_requires_explicit_progress_evidence():
  (pullaway,) = _scenario_result("stopped_lead_pullaway")

  assert pullaway.lead_progress_allowed
  assert pullaway.primary_behavior_lead_idx == 0
  assert pullaway.source == "stop_launch"
  assert pullaway.stack_intent == "launch"
  assert pullaway.a_target > 0.0
  assert not pullaway.should_stop


def test_no_lead_model_stop_can_only_actuate_in_e2e_like_mode():
  (model_stop,) = _scenario_result("no_lead_model_stop")
  (acc_false_stop,) = _scenario_result("false_model_stop_in_acc")

  assert model_stop.requested_mode == "e2e"
  assert model_stop.source == "e2e_stop"
  assert model_stop.should_stop
  assert model_stop.a_target < 0.0

  assert acc_false_stop.requested_mode == "acc"
  assert acc_false_stop.resolved_implementation == "hardware_acc"
  assert acc_false_stop.source == "cruise"
  assert acc_false_stop.selected_candidates == ("cruise:driver_intent:driver_cruise",)


@pytest.mark.parametrize(("scenario", "expected_source"), (
  ("speed_limit_drop", "speed_limit"),
  ("vision_curve_true_positive", "scc_vision"),
  ("downhill_overspeed", "speed_limit"),
))
def test_restrictive_advisories_can_only_lower_accel(scenario, expected_source):
  (frame,) = _scenario_result(scenario)

  assert frame.source == expected_source
  assert frame.stack_intent == "advisory_cap"
  assert frame.a_target < 0.0
  assert all("advisory_increases_accel" not in rejected for rejected in frame.rejected_candidates)


def test_advisory_false_positive_cannot_raise_accel_or_authorize_progress():
  (frame,) = _scenario_result("map_curve_false_positive")

  assert frame.source == "cruise"
  assert frame.stack_intent == "cruise"
  assert frame.a_target == pytest.approx(0.0)
  assert not frame.lead_progress_allowed
  assert any(rejected == "scc_map:advisory_increases_accel" for rejected in frame.rejected_candidates)


def test_one_pedal_lift_off_coasts_without_declaring_stop_hazard():
  (frame,) = _scenario_result("one_pedal_lift_off")

  assert frame.source == "cruise"
  assert frame.stack_reason == "one_pedal_lift_off"
  assert frame.a_target == pytest.approx(0.0)
  assert not frame.should_stop
  assert frame.primary_physical_lead_idx is None
