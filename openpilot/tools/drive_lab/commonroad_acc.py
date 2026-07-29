from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpilot.tools.drive_lab.longitudinal_scenarios import REALISM_MODES, Scenario, _validate_mode

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "commonroad"
BUNDLED_SCENARIOS = (
  "ZAM_ACC-1_1_T-1",
  "ZAM_ACC-1_2_T-1",
  "ZAM_ACC-1_3_T-1",
  "ZAM_CC-1_1_T-1",
)


def generate_commonroad_acc_scenarios(mode: str = "comfort", *, scenario_path: str | None = None) -> list[Scenario]:
  _validate_mode(mode)
  if scenario_path is not None:
    return [_scenario_from_fixture(_load_fixture(Path(scenario_path)), mode)]
  return [_scenario_from_fixture(_load_fixture(FIXTURE_DIR / f"{name}.json"), mode) for name in BUNDLED_SCENARIOS]


def _load_fixture(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as handle:
    return json.load(handle)


def _scenario_from_fixture(data: dict[str, Any], mode: str) -> Scenario:
  kwargs = dict(data["kwargs"])
  return Scenario(
    mode,
    str(data["kind"]),
    str(data["title"]),
    float(data["duration"]),
    kwargs,
    oracle_profile=str(data.get("oracle_profile", "comfort")),
    provenance=dict(data.get("provenance", {})),
  )


def scenario_from_commonroad_xml(path: str, *, mode: str = "comfort") -> Scenario:
  """Optional developer import when commonroad-io is installed."""
  pytest = __import__("pytest")
  commonroad_io = pytest.importorskip("commonroad.common.file_reader")
  from commonroad.common.file_reader import CommonRoadFileReader

  scenario, _ = CommonRoadFileReader(path).open()
  ego = next(o for o in scenario.obstacle_by_id.values() if o.obstacle_role.name == "EGO")
  lead = next(
    (o for o in scenario.obstacle_by_id.values() if o is not ego and o.obstacle_type.name == "CAR"),
    None,
  )
  if lead is None:
    raise ValueError(f"no lead vehicle found in {path}")

  dt = 0.05
  duration = float(scenario.duration)
  times = [i * dt for i in range(int(duration / dt) + 1)]
  ego_states = [ego.state_at_time(t) for t in times]
  lead_states = [lead.state_at_time(t) for t in times]

  v_ego0 = float(ego_states[0].velocity)
  d_rel0 = float(lead_states[0].position[0] - ego_states[0].position[0])
  speed_lead = [float(s.velocity) for s in lead_states]
  return Scenario(
    mode,
    f"commonroad_{Path(path).stem}",
    f"commonroad imported {Path(path).stem}",
    duration,
    {
      "initial_speed": round(v_ego0, 3),
      "lead_relevancy": True,
      "initial_distance_lead": round(max(d_rel0, 5.0), 3),
      "speed_lead_values": [round(v, 3) for v in speed_lead],
      "prob_lead_values": [1.0] * len(speed_lead),
      "cruise_values": [round(v_ego0, 3)] * len(speed_lead),
      "breakpoints": [round(t, 3) for t in times],
    },
    oracle_profile="comfort",
    provenance={"source": "commonroad", "path": path},
  )
