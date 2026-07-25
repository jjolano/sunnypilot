import contextlib
import io
import sys

import numpy as np
import pytest

from openpilot.tools.drive_lab import fuzz_longitudinal
from openpilot.tools.drive_lab.fuzz_longitudinal import (
  Scenario,
  aggregate_mpc_solution_status_counts,
  capture_commanded_accel,
  diagnose_max_jerk,
  evaluate_accordion_response,
  evaluate_collision_response,
  evaluate_invariants,
  evaluate_lead_pullaway_start,
  generate_openpilot_acc_scenarios,
  generate_scenarios,
  generate_udacity_acc_scenarios,
  render_jerk_diagnosis,
  run_scenario,
  scenario_maneuver_kwargs,
  scenario_to_spec,
  shipped_longitudinal_config,
)
from openpilot.tools.drive_lab.log_profile import LongitudinalProfile, ProfileRange
from openpilot.tools.drive_lab.longitudinal_scenarios import (
  SCENARIO_PRESETS,
  generate_preset_scenarios,
  PresetRequest,
)
from openpilot.tools.drive_lab.ncap_acc_scenarios import generate_ncap_acc_scenarios
from openpilot.tools.drive_lab.commonroad_acc import generate_commonroad_acc_scenarios


def test_generate_scenarios_is_seeded():
  assert generate_scenarios(seed=42, cases=5) == generate_scenarios(seed=42, cases=5)


def test_shipped_config_uses_defaults_and_restores_local_overrides(monkeypatch):
  class FakeParams:
    values = {"CustomLongitudinalEnabled": False, "CutInBrakeAssistMode": "apply"}
    defaults = dict.fromkeys(fuzz_longitudinal._SHIPPED_LONGITUDINAL_PARAM_KEYS)
    defaults.update({"CustomLongitudinalEnabled": True, "CutInBrakeAssistMode": "off"})
    writes = []

    def get(self, key):
      return self.values.get(key)

    def get_default_value(self, key):
      return self.defaults[key]

    def put(self, key, value, *, block=False):
      self.writes.append((key, block))
      self.values[key] = value

    def remove(self, key):
      self.values.pop(key, None)

  monkeypatch.setattr("openpilot.common.params.Params", FakeParams)

  with shipped_longitudinal_config():
    assert FakeParams.values == {"CustomLongitudinalEnabled": True, "CutInBrakeAssistMode": "off"}

  assert FakeParams.values == {"CustomLongitudinalEnabled": False, "CutInBrakeAssistMode": "apply"}
  assert FakeParams.writes
  assert all(block for _, block in FakeParams.writes)


def test_comfort_stopped_lead_decel_is_plausible():
  scenarios = [s for s in generate_scenarios(seed=4, cases=100, mode="comfort") if s.kind == "stopped_lead_approach"]

  assert scenarios
  for scenario in scenarios:
    speeds = scenario.kwargs["speed_lead_values"]
    breakpoints = scenario.kwargs["breakpoints"]
    lead_decel = (speeds[1] - speeds[0]) / (breakpoints[1] - breakpoints[0])
    assert -3.6 <= lead_decel <= -1.4


def test_comfort_cut_in_is_not_already_impossible_when_detected():
  scenarios = [s for s in generate_scenarios(seed=5, cases=100, mode="comfort") if s.kind == "slower_cut_in"]

  assert scenarios
  for scenario in scenarios:
    v_ego = scenario.kwargs["initial_speed"]
    v_lead = scenario.kwargs["speed_lead_values"][0]
    cut_in_time = scenario.kwargs["breakpoints"][1]
    detected_gap = scenario.kwargs["initial_distance_lead"] - max(0.0, v_ego - v_lead) * cut_in_time
    assert detected_gap >= max(24.0, v_ego * 1.45)


def test_profile_biases_generated_ranges():
  profile = LongitudinalProfile(
    source="test",
    sample_count=100,
    ego_speed=ProfileRange(12.0, 13.0),
    cruise_speed=ProfileRange(6.0, 7.0),
    lead_gap=ProfileRange(35.0, 40.0),
    closing_speed=ProfileRange(1.0, 1.5),
    lead_decel=ProfileRange(2.0, 2.5),
    stopped_lead_gap=ProfileRange(5.0, 6.0),
    lead_pullaway_speed=ProfileRange(1.5, 2.0),
  )
  scenarios = generate_scenarios(seed=1, cases=100, mode="comfort", profile=profile)

  pullaways = [s for s in scenarios if s.kind == "lead_pullaway"]
  assert pullaways
  for scenario in pullaways:
    assert 5.0 <= scenario.kwargs["initial_distance_lead"] <= 6.0
    assert 1.5 <= scenario.kwargs["speed_lead_values"][2] <= 2.0


def test_udacity_acc_scenarios_cover_all_cases():
  scenarios = generate_udacity_acc_scenarios()

  assert len(scenarios) == 15
  kinds = {scenario.kind for scenario in scenarios}
  assert "udacity_acc_cruise_speed_decrease" in kinds
  assert "udacity_acc_grade_downhill" in kinds
  assert "udacity_acc_approach_from_stop" in kinds
  assert "udacity_acc_accel_while_lead_decel_hard" in kinds
  for scenario in scenarios:
    assert scenario.mode == "comfort"
    assert scenario.title.startswith("udacity acc inspired")
    assert scenario.duration > 0.0
    if "breakpoints" in scenario.kwargs:
      assert scenario.kwargs["breakpoints"] == sorted(scenario.kwargs["breakpoints"])


def test_udacity_lead_decel_stop_uses_regression_oracle():
  scenarios = {scenario.kind: scenario for scenario in generate_udacity_acc_scenarios()}

  assert scenarios["udacity_acc_lead_decel_to_stop"].oracle_profile == "regression"
  assert scenarios["udacity_acc_lead_decel_to_stop_2ms2"].oracle_profile == "regression"
  assert scenarios["udacity_acc_stopped_lead"].oracle_profile == "comfort"


def test_udacity_lead_decel_stop_regression_oracle_passes_comfort_gate():
  scenarios = [
    scenario for scenario in generate_udacity_acc_scenarios()
    if scenario.kind in {"udacity_acc_lead_decel_to_stop", "udacity_acc_lead_decel_to_stop_2ms2"}
  ]

  with shipped_longitudinal_config():
    results = [run_scenario(scenario, max_normal_jerk=8.0) for scenario in scenarios]

  assert [(result.scenario.kind, result.failures) for result in results] == [
    ("udacity_acc_lead_decel_to_stop", []),
    ("udacity_acc_lead_decel_to_stop_2ms2", []),
  ]


def test_udacity_approach_from_stop_passes_comfort_gate():
  scenario = next(s for s in generate_udacity_acc_scenarios() if s.kind == "udacity_acc_approach_from_stop")

  with shipped_longitudinal_config():
    result = run_scenario(scenario, max_normal_jerk=8.0)

  assert result.failures == []


def test_udacity_accordion_cases_do_not_amplify_the_lead_wave():
  scenarios = [
    scenario for scenario in generate_udacity_acc_scenarios()
    if scenario.kind in {"udacity_acc_oscillating_lead", "udacity_acc_stop_and_go_10mph"}
  ]

  with shipped_longitudinal_config():
    results = [run_scenario(scenario, max_normal_jerk=8.0) for scenario in scenarios]

  assert [(result.scenario.kind, result.failures) for result in results] == [
    ("udacity_acc_oscillating_lead", []),
    ("udacity_acc_stop_and_go_10mph", []),
  ]


def test_openpilot_acc_preset_count():
  scenarios = generate_openpilot_acc_scenarios()
  assert len(scenarios) == 15
  assert any(s.kind == "openpilot_resume_from_stop" for s in scenarios)


def test_openpilot_stopped_lead_variants_use_regression_oracle():
  scenarios = {scenario.title: scenario for scenario in generate_openpilot_acc_scenarios()}

  for title in (
    "approach stopped car at 25m/s, initial distance: 120m",
    "approach stopped car at 20m/s, initial distance 90m",
    "approach stopped car at 20m/s, with prob_lead_values",
    "approach stopped car at 20m/s, with prob_throttle_values and pitch = -0.1",
    "approach stopped car at 20m/s, with prob_throttle_values and pitch = +0.1",
  ):
    assert scenarios[title].oracle_profile == "regression"


def test_openpilot_resume_from_stop_uses_launch_oracle():
  scenario = next(s for s in generate_openpilot_acc_scenarios() if s.kind == "openpilot_resume_from_stop")

  assert scenario.oracle_profile == "comfort"
  assert scenario_maneuver_kwargs(scenario)["ensure_start"] is False


def test_openpilot_stopped_lead_prob_variant_passes_regression_gate():
  scenario = next(
    scenario for scenario in generate_openpilot_acc_scenarios()
    if scenario.title == "approach stopped car at 20m/s, with prob_lead_values"
  )

  with shipped_longitudinal_config():
    result = run_scenario(scenario, max_normal_jerk=8.0)

  assert result.failures == []


def test_openpilot_resume_from_stop_passes_launch_gate():
  scenario = next(s for s in generate_openpilot_acc_scenarios() if s.kind == "openpilot_resume_from_stop")

  with shipped_longitudinal_config():
    result = run_scenario(scenario, max_normal_jerk=8.0)

  assert result.failures == []


def test_seeded_pullaway_cases_pass_comfort_jerk_gate():
  scenarios = [
    scenario for scenario in generate_scenarios(seed=1, cases=100, mode="comfort")
    if scenario.title in {"fuzz lead pullaway #20", "fuzz lead pullaway #56", "fuzz lead pullaway #78"}
  ]

  assert len(scenarios) == 3
  with shipped_longitudinal_config():
    results = [run_scenario(scenario, max_normal_jerk=8.0) for scenario in scenarios]

  assert [(result.scenario.title, result.failures) for result in results] == [
    ("fuzz lead pullaway #20", []),
    ("fuzz lead pullaway #56", []),
    ("fuzz lead pullaway #78", []),
  ]


def test_stop_hold_release_step_cases_pass_comfort_jerk_gate():
  # Latch-release frames used to step from the hold command straight to the release accel,
  # bypassing the finalizer's up-jerk slew (release-slew seed). These two cases caught it.
  cases = [(3, "fuzz lead pullaway #91"), (7, "fuzz lead pullaway #55")]

  results = []
  with shipped_longitudinal_config():
    for seed, title in cases:
      scenario = next(s for s in generate_scenarios(seed=seed, cases=100, mode="comfort") if s.title == title)
      results.append(run_scenario(scenario, max_normal_jerk=8.0))

  assert [(result.scenario.title, result.failures) for result in results] == [
    ("fuzz lead pullaway #91", []),
    ("fuzz lead pullaway #55", []),
  ]


def test_ncap_acc_curated_count():
  scenarios = generate_ncap_acc_scenarios()
  assert len(scenarios) == 18
  assert all(s.oracle_profile == "safety" for s in scenarios)


def test_ncap_ccrm_distance_scales_with_closing_speed():
  scenarios = [scenario for scenario in generate_ncap_acc_scenarios() if scenario.kind.startswith("ncap_ccrm_")]

  for scenario in scenarios:
    ego_speed = scenario.kwargs["initial_speed"]
    target_speed = scenario.kwargs["speed_lead_values"][0]
    relative_speed = max(0.0, ego_speed - target_speed)
    assert scenario.kwargs["initial_distance_lead"] >= max(80.0, relative_speed * 4.5)


def test_ncap_fast_ccrm_cases_pass_safety_gate():
  scenarios = [
    scenario for scenario in generate_ncap_acc_scenarios()
    if scenario.kind in {"ncap_ccrm_110_20", "ncap_ccrm_120_20", "ncap_ccrm_130_20"}
  ]

  with shipped_longitudinal_config():
    results = [run_scenario(scenario, max_normal_jerk=8.0) for scenario in scenarios]

  assert [(result.scenario.kind, result.failures) for result in results] == [
    ("ncap_ccrm_110_20", []),
    ("ncap_ccrm_120_20", []),
    ("ncap_ccrm_130_20", []),
  ]


def test_ncap_acc_sample_family():
  scenarios = generate_ncap_acc_scenarios(mode="comfort", family="CCRs", sample=2, seed=1)
  assert len(scenarios) == 2
  assert all(s.kind.startswith("ncap_ccrs_") for s in scenarios)


def test_commonroad_acc_fixtures():
  scenarios = generate_commonroad_acc_scenarios()
  assert len(scenarios) == 4
  assert scenarios[0].kind.startswith("commonroad_")


def test_all_presets_registered():
  assert set(SCENARIO_PRESETS) == {"fuzz", "udacity-acc", "openpilot-acc", "ncap-acc", "commonroad-acc",
                                     "nuscenes-acc", "iso15622-acc", "unr157-alks", "nhtsa-fcw", "cncap-ccrh", "iihs-acc"}


def test_main_lists_udacity_acc_preset():
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal.py", "--preset", "udacity-acc", "--list-only"]
    with contextlib.redirect_stdout(stdout):
      fuzz_longitudinal.main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "udacity acc inspired green light lead launch" in output
  assert "fuzz stopped lead approach" not in output


def test_evaluate_invariants_catches_collision_and_nan():
  output = np.array([
    [0.0, 0.0, 0.0, 10.0, 9.0, 0.0, 10.0],
    [0.1, 1.0, 1.0, 10.0, 9.0, np.nan, 0.2],
  ])

  failures = evaluate_invariants(True, output)

  assert [f.check for f in failures] == ["finite"]


def test_evaluate_invariants_reports_malformed_output_shape():
  failures = evaluate_invariants(True, np.zeros((2, 3)))
  assert any(f.check == "output" for f in failures)


def test_accordion_oracle_rejects_speed_wave_amplification():
  output = np.zeros((5, 7))
  output[:, 3] = [5.0, 0.0, 7.0, 0.0, 5.0]  # ego speed
  output[:, 4] = [5.0, 0.0, 5.0, 0.0, 5.0]  # lead speed

  failures = evaluate_accordion_response(output)

  assert len(failures) == 1
  assert failures[0].check == "accordion"
  assert "gain 1.200" in failures[0].detail


def test_accordion_oracle_accepts_manual_style_attenuation():
  output = np.zeros((5, 7))
  output[:, 3] = [5.0, 0.5, 4.5, 0.5, 4.5]  # attenuated speed variation
  output[:, 4] = [5.0, 0.0, 5.0, 0.0, 5.0]

  assert evaluate_accordion_response(output) == []


def test_evaluate_lead_pullaway_start_detects_no_launch():
  output = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 9.0],
    [2.0, 2.0, 2.0, 0.0, 1.0, 0.0, 8.0],
  ])
  failures = evaluate_lead_pullaway_start(output)
  assert failures and failures[0].check == "launch"


def _closing_trace(n: int) -> np.ndarray:
  # n frames closing from 10 m to contact, ego 8 -> 4 m/s against a stopped lead
  d_rel = np.linspace(10.0, 0.3, n)
  v_ego = np.linspace(8.0, 4.0, n)
  out = np.zeros((n, 7))
  out[:, 0] = np.arange(n) * 0.05
  out[:, 3] = v_ego
  out[:, 4] = 0.0
  out[:, 5] = -3.0
  out[:, 6] = d_rel
  return out


def test_evaluate_collision_response_accepts_sustained_best_effort_brake():
  n = 20  # 1.0 s at DT_MDL, comfortably over BEST_EFFORT_MIN_S
  output = _closing_trace(n)
  commanded = np.full(n, -3.0)
  prob_lead = np.ones(n)
  assert not evaluate_collision_response(output, commanded, prob_lead)


def test_evaluate_collision_response_rejects_single_frame_brake_dip():
  # One frame at the brake threshold is not a best effort: a hard collision must
  # not be excused by a momentary dip.
  n = 20
  output = _closing_trace(n)
  commanded = np.full(n, -0.2)
  commanded[5] = -3.0
  failures = evaluate_collision_response(output, commanded, np.ones(n))
  assert failures and failures[0].check == "collision"


def test_evaluate_collision_response_safety_profile_fails_high_impact():
  output = np.array([
    [0.0, 0.0, 0.0, 10.0, 0.0, -1.0, 10.0],
    [0.1, 1.0, 1.0, 8.0, 0.0, -1.0, 0.3],
  ])
  failures = evaluate_collision_response(output, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), max_impact_speed_ms=1.0, use_best_effort=False)
  assert failures


def test_scenario_maneuver_kwargs_disables_ensure_start_for_launch_oracle():
  scenario = generate_udacity_acc_scenarios()[next(i for i, s in enumerate(generate_udacity_acc_scenarios()) if s.kind == "udacity_acc_green_light_launch")]
  kwargs = scenario_maneuver_kwargs(scenario)
  assert kwargs["ensure_start"] is False


def test_scenario_to_spec_includes_oracle_profile_in_tags():
  scenario = generate_udacity_acc_scenarios()[0]
  spec = scenario_to_spec(scenario, source="udacity-acc")
  assert "udacity-acc" in spec.tags


def test_preset_request_openpilot():
  scenarios = generate_preset_scenarios(PresetRequest(preset="openpilot-acc", cases=1))
  assert len(scenarios) == 15


def test_run_scenario_smoke():
  scenario = generate_scenarios(seed=1, cases=1)[0]
  with shipped_longitudinal_config():
    result = run_scenario(scenario)
  assert isinstance(result.valid, bool)


def test_aggregate_mpc_solution_status_counts_sums_per_status():
  dummy = Scenario(mode="comfort", kind="test", title="dummy", duration=1.0, kwargs={})
  results = [
    fuzz_longitudinal.ScenarioResult(scenario=dummy, valid=True, failures=[], mpc_solution_status_counts={4: 2, 5: 1}),
    fuzz_longitudinal.ScenarioResult(scenario=dummy, valid=True, failures=[], mpc_solution_status_counts={4: 1}),
    fuzz_longitudinal.ScenarioResult(scenario=dummy, valid=True, failures=[], mpc_solution_status_counts={}),
  ]
  assert aggregate_mpc_solution_status_counts(results) == {4: 3, 5: 1}


def test_capture_commanded_accel_records_mpc_reset_status():
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
  with capture_commanded_accel() as capture:
    mpc = LongitudinalMpc()
    mpc.solution_status = 4
    mpc.reset()
    mpc.solution_status = 4
    mpc.reset()
    mpc.solution_status = 2
    mpc.reset()
  assert capture.mpc_solution_status_counts == {4: 2, 2: 1}


def test_main_json_includes_mpc_resets(monkeypatch):
  import json
  scenarios = [Scenario(mode="comfort", kind="test", title="test scenario", duration=1.0, kwargs={})]

  def fake_run_scenario(scenario, max_normal_jerk=8.0):
    return fuzz_longitudinal.ScenarioResult(
      scenario=scenario,
      valid=True,
      failures=[],
      mpc_solution_status_counts={4: 3, 2: 1},
    )

  monkeypatch.setattr(fuzz_longitudinal, "generate_preset_scenarios", lambda request: scenarios)
  monkeypatch.setattr(fuzz_longitudinal, "run_scenario", fake_run_scenario)

  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal.py", "--preset", "fuzz", "--cases", "1", "--json"]
    with contextlib.redirect_stdout(stdout):
      fuzz_longitudinal.main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  assert payload["mpcSolutionStatusCounts"] == {"4": 3, "2": 1}
  assert payload["totalMpcResets"] == 4
  assert payload["scenarioResults"][0]["mpcSolutionStatusCounts"] == {"4": 3, "2": 1}
  assert payload["failures"] == []


def test_diagnose_max_jerk_identifies_peak_window_and_slew_call():
  frames = [
    fuzz_longitudinal.CommandFrame(idx=0, time_s=0.0, a_cmd=0.0, a_plant=0.0, v_ego=0.0, v_lead=0.0, d_rel=10.0, prob_lead=1.0, output_should_stop=False),
    fuzz_longitudinal.CommandFrame(idx=1, time_s=0.1, a_cmd=0.0, a_plant=0.0, v_ego=0.0, v_lead=0.0, d_rel=10.0, prob_lead=1.0, output_should_stop=False),
    fuzz_longitudinal.CommandFrame(idx=2, time_s=0.2, a_cmd=2.0, a_plant=0.0, v_ego=0.0, v_lead=0.0, d_rel=10.0, prob_lead=1.0, output_should_stop=False),
    fuzz_longitudinal.CommandFrame(idx=3, time_s=0.3, a_cmd=2.0, a_plant=0.0, v_ego=0.0, v_lead=0.0, d_rel=10.0, prob_lead=1.0, output_should_stop=False),
  ]
  slew_calls = [
    fuzz_longitudinal.SlewCall(0, 0.0, 0.0, False),
    fuzz_longitudinal.SlewCall(1, 0.0, 0.0, False),
    fuzz_longitudinal.SlewCall(2, 2.0, 1.0, True),
    fuzz_longitudinal.SlewCall(3, 2.0, 2.0, False),
  ]
  diag = diagnose_max_jerk(frames, jerk_window=1, slew_calls=slew_calls)
  assert diag is not None
  assert diag.idx0 == 1
  assert diag.idx1 == 2
  assert diag.jerk == pytest.approx(20.0)
  assert diag.slew_input == pytest.approx(2.0)
  assert diag.slew_output == pytest.approx(1.0)
  assert diag.slew_capped is True


def test_render_jerk_diagnosis_includes_key_fields():
  frame = fuzz_longitudinal.CommandFrame(
    idx=2, time_s=0.2, a_cmd=2.0, a_plant=0.0, v_ego=0.5, v_lead=1.0, d_rel=8.0, prob_lead=1.0,
    output_should_stop=False,
    debug={"final_should_stop": True},
    custom={
      "release_block_reason": "custom_should_stop",
      "lead_stop_hold_active": True,
      "standstill_release_source": "lead_pullaway",
      "standstill_release_allowed": True,
      "stop_hold_release_slew_a_target": 0.1,
    },
  )
  diag = fuzz_longitudinal.JerkDiagnosis(
    idx0=1, idx1=2, time0=0.1, time1=0.2, dt=0.1, jerk_window=1,
    a0=0.0, a1=2.0, delta_a=2.0, jerk=20.0, frames=[frame],
    slew_input=2.0, slew_output=1.0, slew_capped=True,
  )
  text = render_jerk_diagnosis(diag)
  assert "jerk diagnosis:" in text
  assert "jerk=20.000" in text
  assert "should_stop=True" in text
  assert "custom_should_stop" in text
  assert "lead_stop_hold=True" in text
  assert "2.000->1.000 capped=True" in text


def _make_jerk_scenario_result(scenario, jerk=100.0):
  diag = fuzz_longitudinal.JerkDiagnosis(
    idx0=0, idx1=1, time0=0.0, time1=0.1, dt=0.1, jerk_window=1,
    a0=0.0, a1=10.0, delta_a=10.0, jerk=jerk, frames=[],
    slew_input=10.0, slew_output=5.0, slew_capped=True,
  )
  return fuzz_longitudinal.ScenarioResult(
    scenario=scenario,
    valid=False,
    failures=[fuzz_longitudinal.ScenarioFailure("jerk", f"maximum absolute jerk {jerk:.3f} m/s^3")],
    mpc_solution_status_counts={},
    jerk_diagnosis=diag,
  )


def test_main_json_includes_jerk_diagnosis(monkeypatch):
  import json
  scenario = Scenario(mode="comfort", kind="test", title="jerk scenario", duration=1.0, kwargs={})

  monkeypatch.setattr(fuzz_longitudinal, "generate_preset_scenarios", lambda request: [scenario])
  monkeypatch.setattr(fuzz_longitudinal, "run_scenario", lambda s, max_normal_jerk=8.0: _make_jerk_scenario_result(s))

  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal.py", "--preset", "fuzz", "--cases", "1", "--json"]
    with contextlib.redirect_stdout(stdout):
      with pytest.raises(SystemExit):
        fuzz_longitudinal.main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  failure = payload["failures"][0]
  assert failure["checks"][0]["check"] == "jerk"
  assert failure["jerkDiagnosis"]["jerk"] == pytest.approx(100.0)
  assert failure["jerkDiagnosis"]["slewCapped"] is True


def test_main_text_includes_jerk_diagnosis(monkeypatch):
  scenario = Scenario(mode="comfort", kind="test", title="jerk scenario", duration=1.0, kwargs={})

  monkeypatch.setattr(fuzz_longitudinal, "generate_preset_scenarios", lambda request: [scenario])
  monkeypatch.setattr(fuzz_longitudinal, "run_scenario", lambda s, max_normal_jerk=8.0: _make_jerk_scenario_result(s))

  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal.py", "--preset", "fuzz", "--cases", "1"]
    with contextlib.redirect_stdout(stdout):
      with pytest.raises(SystemExit):
        fuzz_longitudinal.main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "jerk diagnosis:" in output
  assert "should_stop=" in output
  assert "slew=" in output


def _make_jerk_frame_with_gate():
  custom = {
    "selected_lead_id": 7,
    "lead_stop_hold_lead_id": 7,
    "lead_stop_hold_active": False,
    "lead_stop_hold_gap_increasing_s": 0.25,
    "lead_stop_hold_gap_baseline_d_rel": 5.7,
    "standstill_release_confidence_mode": "gate",
    "custom_long_enabled": True,
    "custom_output_enabled": True,
    "custom_should_stop": False,
    "standstill_release_source": "lead_pullaway",
    "standstill_release_allowed": True,
    "same_id": True,
    "baseline_opening": 0.5,
    "prep_applies": False,
    "prep_gate_would_apply": True,
    "prep_block_reason": "not_hold_branch",
    "release_path": "standstill_release_clear",
    "latch_reset_on_frame": False,
    "release_min_d_rel": 6.2,
    "prep_min_d_rel": 6.2,
    "d_rel_minus_release_min_d_rel": 0.0,
    "d_rel_minus_prep_min_d_rel": 0.0,
  }
  debug = {"mpc_a_target": 0.2, "model_a_target": 0.1, "model_should_stop": False}
  return fuzz_longitudinal.CommandFrame(
    idx=5, time_s=0.25, a_cmd=0.0, a_plant=0.0, v_ego=0.16, v_lead=1.93, d_rel=6.2,
    prob_lead=1.0, output_should_stop=False, debug=debug, custom=custom,
  )


def test_frame_release_gate_context_computes_release_and_prep_min():
  custom = {
    "selected_lead_id": 7,
    "lead_stop_hold_lead_id": 7,
    "lead_stop_hold_active": False,
    "lead_stop_hold_gap_increasing_s": 0.25,
    "lead_stop_hold_gap_baseline_d_rel": 5.7,
    "standstill_release_confidence_mode": "gate",
    "custom_long_enabled": True,
    "custom_output_enabled": True,
    "custom_should_stop": False,
    "standstill_release_source": "lead_pullaway",
    "standstill_release_allowed": True,
  }
  debug = {"mpc_a_target": 0.2, "model_a_target": 0.1, "model_should_stop": False}
  frame = fuzz_longitudinal.CommandFrame(
    idx=5, time_s=0.25, a_cmd=0.0, a_plant=0.0, v_ego=0.0, v_lead=1.0, d_rel=6.25,
    prob_lead=1.0, output_should_stop=False, debug=debug, custom=custom,
  )
  prev = fuzz_longitudinal.CommandFrame(
    idx=4, time_s=0.20, a_cmd=-2.0, a_plant=-2.0, v_ego=0.0, v_lead=0.0, d_rel=6.0,
    prob_lead=1.0, output_should_stop=False, custom={"lead_stop_hold_active": True},
  )
  ctx = fuzz_longitudinal._frame_release_gate_context(frame, prev, 6.0)
  assert ctx["same_id"] is True
  assert ctx["latch_reset_on_frame"] is True
  assert ctx["release_path"] == "lead_stop_hold_release"
  assert ctx["baseline_opening"] == pytest.approx(0.55)
  assert ctx["release_min_d_rel"] == pytest.approx(6.2)
  assert ctx["prep_min_d_rel"] == pytest.approx(6.2)
  assert ctx["prep_applies"] is False
  assert ctx["prep_gate_would_apply"] is True
  assert ctx["prep_block_reason"] == "not_hold_branch"


def test_render_jerk_diagnosis_shows_gate_context():
  frame = _make_jerk_frame_with_gate()
  diag = fuzz_longitudinal.JerkDiagnosis(
    idx0=4, idx1=5, time0=0.20, time1=0.25, dt=0.05, jerk_window=1,
    a0=-2.0, a1=0.0, delta_a=2.0, jerk=40.0, frames=[frame],
  )
  text = render_jerk_diagnosis(diag)
  assert "prep=False" in text
  assert "prepWouldApply" in text
  assert "prepBlock=not_hold_branch" in text
  assert "dRel=6.20" in text
  assert "prepMin=6.20" in text
  assert "releaseMin=6.20" in text
  assert "path=standstill_release_clear" in text
  assert "leadId=7" in text
  assert "latchId=7" in text


def test_main_json_includes_gate_fields(monkeypatch):
  import json
  scenario = Scenario(mode="comfort", kind="test", title="jerk gate scenario", duration=1.0, kwargs={})
  frame = _make_jerk_frame_with_gate()
  diag = fuzz_longitudinal.JerkDiagnosis(
    idx0=4, idx1=5, time0=0.20, time1=0.25, dt=0.05, jerk_window=1,
    a0=-2.0, a1=0.0, delta_a=2.0, jerk=40.0, frames=[frame],
    slew_input=-2.0, slew_output=-2.0, slew_capped=False,
  )

  def fake_run_scenario(s, max_normal_jerk=8.0):
    return fuzz_longitudinal.ScenarioResult(
      scenario=s,
      valid=False,
      failures=[fuzz_longitudinal.ScenarioFailure("jerk", "maximum absolute jerk 40.000 m/s^3")],
      mpc_solution_status_counts={},
      jerk_diagnosis=diag,
    )

  monkeypatch.setattr(fuzz_longitudinal, "generate_preset_scenarios", lambda request: [scenario])
  monkeypatch.setattr(fuzz_longitudinal, "run_scenario", fake_run_scenario)

  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["fuzz_longitudinal.py", "--preset", "fuzz", "--cases", "1", "--json"]
    with contextlib.redirect_stdout(stdout):
      with pytest.raises(SystemExit):
        fuzz_longitudinal.main()
  finally:
    sys.argv = previous_argv

  payload = json.loads(stdout.getvalue())
  failure = payload["failures"][0]
  diag_dict = failure["jerkDiagnosis"]
  assert diag_dict["frames"][0]["custom"]["prepApplies"] is False
  assert diag_dict["frames"][0]["custom"]["prepGateWouldApply"] is True
  assert diag_dict["frames"][0]["custom"]["releasePath"] == "standstill_release_clear"
  assert diag_dict["frames"][0]["custom"]["latchResetOnFrame"] is False
  assert diag_dict["frames"][0]["custom"]["dRelMinusPrepMinDRel"] == pytest.approx(0.0)
