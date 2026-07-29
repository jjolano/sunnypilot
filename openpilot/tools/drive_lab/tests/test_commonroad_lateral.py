import json
from pathlib import Path

from openpilot.tools.drive_lab.commonroad_lateral import (
    BUNDLED_SCENARIOS,
    FIXTURE_DIR,
    generate_commonroad_lateral_scenarios,
    load_commonroad_lateral_fixtures,
)
from openpilot.tools.drive_lab.fuzz_lateral_demand import DemandScenario, evaluate_scenario
from openpilot.tools.drive_lab.lateral_scenarios import (
    LATERAL_PRESETS,
    LateralPresetRequest,
    generate_preset_scenarios,
)


def test_fixture_files_exist():
    for name in BUNDLED_SCENARIOS:
        assert (FIXTURE_DIR / f"{name}.json").is_file()


def test_fixture_json_is_valid_demand_scenario():
    for path in FIXTURE_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "kind" in data
        assert "title" in data
        assert "duration_s" in data
        assert "frames" in data
        assert len(data["frames"]) > 0


def test_load_commonroad_lateral_fixtures():
    scenarios = load_commonroad_lateral_fixtures()
    assert len(scenarios) == len(BUNDLED_SCENARIOS)
    for scenario in scenarios:
        assert isinstance(scenario, DemandScenario)
        assert scenario.kind.startswith("commonroad_lateral_")


def test_each_fixture_evaluates_successfully():
    for scenario in generate_commonroad_lateral_scenarios():
        result = evaluate_scenario(scenario)
        assert result.valid, f"{scenario.kind} failed: {result.failures}"
        assert len(result.outputs) == len(scenario.frames)


def test_commonroad_lateral_preset_generates_four_scenarios():
    assert "commonroad-lateral" in LATERAL_PRESETS
    request = LateralPresetRequest(preset="commonroad-lateral")
    scenarios = generate_preset_scenarios(request)
    assert len(scenarios) == 4
    kinds = {s.kind for s in scenarios}
    expected = {f"commonroad_lateral_{name}" for name in BUNDLED_SCENARIOS}
    assert kinds == expected
