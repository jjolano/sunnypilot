from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openpilot.tools.drive_lab.fuzz_lateral_demand import DemandScenario

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "commonroad_lateral"
BUNDLED_SCENARIOS = (
    "straight_lane_follow",
    "tight_curve",
    "lane_merge",
    "degraded_markings",
)


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _scenario_from_fixture(data: dict[str, Any]) -> DemandScenario:
    from openpilot.tools.drive_lab.fuzz_lateral_demand import scenario_from_dict
    return scenario_from_dict(data)


def load_commonroad_lateral_fixtures() -> list[DemandScenario]:
    """Load all CommonRoad lateral benchmark fixtures as ``DemandScenario`` objects."""
    scenarios: list[DemandScenario] = []
    for name in BUNDLED_SCENARIOS:
        path = FIXTURE_DIR / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing CommonRoad lateral fixture: {path}")
        scenarios.append(_scenario_from_fixture(_load_fixture(path)))
    return scenarios


def generate_commonroad_lateral_scenarios() -> list[DemandScenario]:
    """Generate the CommonRoad lateral benchmark preset scenarios."""
    return load_commonroad_lateral_fixtures()
