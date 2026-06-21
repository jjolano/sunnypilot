import json

import pytest

from openpilot.tools.drive_lab.nuscenes_acc import generate_nuscenes_acc_scenarios


def _write_fixture(tmp_path, samples, **extra):
  path = tmp_path / "nuscenes_sample.json"
  data = {
    "title": "nuscenes sample follow",
    "kind": "nuscenes_sample_follow",
    **extra,
    "samples": samples,
  }
  path.write_text(json.dumps(data), encoding="utf-8")
  return path


def test_nuscenes_from_d_rel(tmp_path):
  path = _write_fixture(
    tmp_path,
    samples=[
      {"t": 5.0, "ego_vx": 15.0, "lead_vx": 14.0, "d_rel": 40.0},
      {"t": 5.2, "ego_vx": 15.5, "lead_vx": 13.0, "d_rel": 39.8},
      {"t": 5.4, "ego_vx": 16.0, "lead_vx": 12.0, "d_rel": 39.5},
    ],
  )

  scenarios = generate_nuscenes_acc_scenarios("comfort", scenario_path=str(path))
  assert len(scenarios) == 1
  scenario = scenarios[0]
  kwargs = scenario.kwargs

  assert scenario.mode == "comfort"
  assert scenario.kind == "nuscenes_sample_follow"
  assert scenario.title == "nuscenes sample follow"
  assert pytest.approx(scenario.duration) == 0.4

  assert kwargs["initial_speed"] == 15.0
  assert kwargs["lead_relevancy"] is True
  assert kwargs["initial_distance_lead"] == 40.0
  assert kwargs["breakpoints"] == pytest.approx([0.0, 0.2, 0.4])
  assert kwargs["speed_lead_values"] == pytest.approx([14.0, 13.0, 12.0])
  assert kwargs["prob_lead_values"] == [1.0, 1.0, 1.0]
  assert kwargs["cruise_values"] == pytest.approx([15.0, 15.5, 16.0])
  assert kwargs["pitch_values"] == pytest.approx([0.0, 0.0, 0.0])


def test_nuscenes_from_positions(tmp_path):
  path = _write_fixture(
    tmp_path,
    samples=[
      {"t": 0.0, "ego_vx": 10.0, "lead_vx": 8.0, "ego_x": 0.0, "lead_x": 50.0},
      {"t": 0.1, "ego_vx": 10.0, "lead_vx": 8.0, "ego_x": 1.0, "lead_x": 50.8},
    ],
  )

  scenarios = generate_nuscenes_acc_scenarios("emergency", scenario_path=str(path))
  assert len(scenarios) == 1
  kwargs = scenarios[0].kwargs

  assert scenarios[0].mode == "emergency"
  assert kwargs["initial_speed"] == 10.0
  assert kwargs["initial_distance_lead"] == 50.0
  assert kwargs["breakpoints"] == pytest.approx([0.0, 0.1])
  assert kwargs["speed_lead_values"] == pytest.approx([8.0, 8.0])


def test_nuscenes_missing_scenario_path():
  with pytest.raises(TypeError):
    generate_nuscenes_acc_scenarios("comfort")  # type: ignore[call-arg]


def test_nuscenes_invalid_mode(tmp_path):
  path = _write_fixture(tmp_path, samples=[{"t": 0.0, "ego_vx": 5.0, "lead_vx": 5.0, "d_rel": 20.0}])
  with pytest.raises(ValueError):
    generate_nuscenes_acc_scenarios("invalid", scenario_path=str(path))


def test_nuscenes_provenance(tmp_path):
  path = _write_fixture(
    tmp_path,
    samples=[{"t": 0.0, "ego_vx": 5.0, "lead_vx": 5.0, "d_rel": 20.0}],
    provenance={"scene": "scene-0061", "token": "abc"},
  )
  scenarios = generate_nuscenes_acc_scenarios("comfort", scenario_path=str(path))
  assert scenarios[0].provenance == {
    "scene": "scene-0061",
    "token": "abc",
    "source": "nuscenes",
    "path": str(path),
  }
