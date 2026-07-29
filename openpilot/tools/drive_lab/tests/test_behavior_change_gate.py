import json

import pytest

from openpilot.tools.drive_lab import behavior_change_gate as gate
from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec


def spec(kind="lead_risk", tags=("route-derived", "longitudinal"), events=("lead_risk",), checks=("manual_agreement",)):
  return ScenarioSpec(
    scenario_id="s1",
    kind=kind,
    title="title",
    mode="route-derived",
    duration=1.0,
    source="src",
    maneuver_kwargs={},
    events=events,
    oracle={"checks": checks},
    tags=tags,
  )


def test_empty_specs_not_ready():
  result = gate.assess_behavior_change_readiness([], "any")

  assert not result.ready
  assert result.reasons == ("no_scenarios",)


def test_longitudinal_route_derived_ready_for_longitudinal_and_any_not_lateral():
  s = spec(tags=("route-derived", "longitudinal"), events=("lead_risk",), checks=("manual_agreement",))

  long_result = gate.assess_behavior_change_readiness([s], "longitudinal")
  any_result = gate.assess_behavior_change_readiness([s], "any")
  lateral_result = gate.assess_behavior_change_readiness([s], "lateral")

  assert long_result.ready and long_result.matching_count == 1
  assert any_result.ready and any_result.matching_count == 1
  assert not lateral_result.ready
  assert lateral_result.reasons == ("no_route_derived_lateral_specs",)


def test_lateral_route_derived_ready_for_lateral():
  s = spec(kind="fast_reversal", tags=("route-derived", "lateral"), events=("fast_reversal",), checks=("finite", "lag"))

  result = gate.assess_behavior_change_readiness([s], "lateral")

  assert result.ready
  assert result.matching_count == 1


def test_missing_events_and_checks_block_readiness():
  s = spec(tags=("route-derived", "longitudinal"), events=(), checks=())

  result = gate.assess_behavior_change_readiness([s], "longitudinal")

  assert not result.ready
  assert result.reasons == ("missing_oracle_checks", "missing_events")


def test_load_scenario_specs_handles_list_and_object_payload(tmp_path):
  scenario = spec().to_dict()
  list_path = tmp_path / "list.json"
  object_path = tmp_path / "object.json"
  list_path.write_text(json.dumps([scenario]))
  object_path.write_text(json.dumps({"scenarios": [scenario]}))

  assert gate.load_scenario_specs(list_path) == [ScenarioSpec.from_dict(scenario)]
  assert gate.load_scenario_specs(object_path) == [ScenarioSpec.from_dict(scenario)]


def test_cli_exit_codes_for_ready_and_not_ready(tmp_path, monkeypatch):
  ready_path = tmp_path / "ready.json"
  not_ready_path = tmp_path / "not_ready.json"
  ready_path.write_text(json.dumps([spec(tags=("route-derived", "lateral"), events=("fast_reversal",), checks=("finite", "lag")).to_dict()]))
  not_ready_path.write_text(json.dumps([spec(tags=("route-derived", "lateral"), events=(), checks=()).to_dict()]))

  monkeypatch.setattr(gate.sys, "argv", ["prog", str(ready_path), "--domain", "lateral"])
  with pytest.raises(SystemExit) as exc:
    gate.main()
  assert exc.value.code == 0

  monkeypatch.setattr(gate.sys, "argv", ["prog", str(not_ready_path), "--domain", "lateral"])
  with pytest.raises(SystemExit) as exc:
    gate.main()
  assert exc.value.code == 1
