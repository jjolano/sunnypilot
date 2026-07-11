import contextlib
import io
import sys
from types import SimpleNamespace

from openpilot.selfdrive.test.longitudinal_maneuvers import maneuver as maneuver_module

from openpilot.tools.drive_lab import compare_manual_longitudinal_baseline as baseline_cli
from openpilot.tools.drive_lab.manual_longitudinal_baseline import (
  ExpectedRange,
  MetricComparison,
  ScenarioComparison,
)


def test_cli_prints_synthetic_baseline_report(monkeypatch):
  scenario = SimpleNamespace(kind="lead_pullaway", title="synthetic lead pullaway", duration=3.0, kwargs={}, oracle_profile="comfort")
  comparison = MetricComparison("Launch", "launch_delay", "launch delay", 0.4, ExpectedRange(0.0, 1.2, "s"), True)
  monkeypatch.setattr(baseline_cli, "generate_preset_scenarios", lambda request: [scenario])
  monkeypatch.setattr(baseline_cli, "load_profile", lambda path: None)
  monkeypatch.setattr(
    baseline_cli,
    "evaluate_scenario",
    lambda selected: ScenarioComparison(selected.title, selected.kind, True, [comparison]),
  )
  stdout = io.StringIO()
  previous_argv = sys.argv
  try:
    sys.argv = ["compare_manual_longitudinal_baseline.py", "--seed", "7", "--cases", "1"]
    with contextlib.redirect_stdout(stdout):
      baseline_cli.main()
  finally:
    sys.argv = previous_argv

  output = stdout.getvalue()
  assert "Drive Lab manual longitudinal baseline" in output
  assert "synthetic lead pullaway" in output
  assert "| Area | Metric | Current | Expected | Result |" in output


def test_cli_strict_exits_when_comparison_fails(monkeypatch):
  scenario = SimpleNamespace(kind="lead_pullaway", title="synthetic lead pullaway", duration=3.0, kwargs={}, oracle_profile="comfort")
  comparison = MetricComparison("Launch", "launch_delay", "launch delay", 2.0, ExpectedRange(0.0, 1.2, "s"), False)
  monkeypatch.setattr(baseline_cli, "generate_preset_scenarios", lambda request: [scenario])
  monkeypatch.setattr(baseline_cli, "load_profile", lambda path: None)
  monkeypatch.setattr(
    baseline_cli,
    "evaluate_scenario",
    lambda selected: ScenarioComparison(selected.title, selected.kind, True, [comparison]),
  )
  previous_argv = sys.argv
  try:
    sys.argv = ["compare_manual_longitudinal_baseline.py", "--cases", "1", "--strict"]
    with contextlib.redirect_stdout(io.StringIO()):
      try:
        baseline_cli.main()
      except SystemExit as exc:
        assert exc.code == 1
      else:
        raise AssertionError("strict baseline should exit when a comparison fails")
  finally:
    sys.argv = previous_argv


def test_evaluate_scenario_uses_bounded_lead_pullaway_start_oracle(monkeypatch):
  captured_kwargs = {}

  class FakeManeuver:
    def __init__(self, title, duration, **kwargs):
      captured_kwargs.update(kwargs)

    def evaluate(self):
      return True, []

  scenario = SimpleNamespace(
    kind="lead_pullaway",
    title="synthetic lead pullaway",
    duration=3.0,
    kwargs={"ensure_start": True},
    oracle_profile="comfort",
  )
  monkeypatch.setattr(maneuver_module, "Maneuver", FakeManeuver)
  monkeypatch.setattr(baseline_cli, "compare_scenario_output", lambda kind, output, **kwargs: [])

  baseline_cli.evaluate_scenario(scenario)

  assert captured_kwargs["ensure_start"] is False
