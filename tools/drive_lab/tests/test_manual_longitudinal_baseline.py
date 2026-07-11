import numpy as np

from openpilot.tools.drive_lab.manual_longitudinal_baseline import (
  compare_scenario_output,
  render_behavior_outline,
  render_comparison_table,
)


def test_lead_pullaway_comparison_reports_launch_delay_and_table():
  output = np.array([
    [0.0, 0.0, 6.0, 0.00, 0.0, 0.00, 6.0],
    [1.0, 0.0, 6.0, 0.00, 0.8, 0.00, 6.8],
    [1.4, 0.1, 6.4, 0.25, 1.2, 0.55, 7.0],
    [2.0, 0.5, 7.0, 0.90, 1.2, 0.65, 7.1],
    [3.0, 2.0, 8.2, 2.10, 1.2, 0.70, 7.2],
  ])

  comparisons = compare_scenario_output("lead_pullaway", output)
  by_metric = {comparison.metric: comparison for comparison in comparisons}

  assert by_metric["launch_delay"].current == 0.4
  assert by_metric["launch_delay"].passed
  assert by_metric["launch_peak_accel"].passed

  table = render_comparison_table(comparisons)
  assert "| Area | Metric | Current | Expected | Result |" in table
  assert "Launch" in table
  assert "launch delay" in table


def test_lead_pullaway_allows_gentle_human_like_launch_mean():
  output = np.array([
    [0.0, 0.0, 6.0, 0.00, 0.0, 0.00, 6.0],
    [1.0, 0.0, 6.0, 0.00, 0.8, 0.00, 6.8],
    [1.3, 0.1, 6.4, 0.25, 1.0, 0.20, 7.0],
    [2.0, 0.5, 7.0, 0.90, 1.0, 0.24, 7.1],
    [3.0, 2.0, 8.0, 2.10, 1.0, 0.25, 7.2],
  ])

  comparisons = compare_scenario_output("lead_pullaway", output)
  by_metric = {comparison.metric: comparison for comparison in comparisons}

  assert by_metric["launch_mean_accel"].current == 0.23
  assert by_metric["launch_mean_accel"].passed


def test_lead_pullaway_launch_mean_uses_initial_launch_not_settled_follow():
  output = np.array([
    [0.0, 0.0, 6.0, 0.00, 0.0, 0.00, 6.0],
    [1.0, 0.0, 6.0, 0.00, 1.0, 0.00, 6.8],
    [1.2, 0.1, 6.4, 0.25, 1.0, 0.50, 7.0],
    [2.0, 0.5, 7.0, 0.90, 1.0, 0.50, 7.1],
    [4.5, 2.2, 8.0, 1.30, 1.0, -0.10, 7.2],
    [6.5, 4.0, 10.0, 1.20, 1.0, -0.10, 7.4],
  ])

  comparisons = compare_scenario_output("lead_pullaway", output)
  by_metric = {comparison.metric: comparison for comparison in comparisons}

  assert by_metric["launch_mean_accel"].current == 0.5
  assert by_metric["launch_mean_accel"].passed


def test_stopped_lead_approach_flags_hard_late_stop_and_short_gap():
  output = np.array([
    [0.0, 0.0, 40.0, 10.0, 8.0, 0.0, 40.0],
    [1.0, 9.0, 48.0, 8.0, 4.0, -0.8, 39.0],
    [2.0, 15.0, 50.0, 4.0, 0.0, -1.2, 35.0],
    [3.0, 18.0, 50.0, 0.8, 0.0, -3.0, 2.5],
    [4.0, 18.1, 50.0, 0.0, 0.0, 0.0, 2.4],
  ])

  comparisons = compare_scenario_output("stopped_lead_approach", output)
  failed = {comparison.metric for comparison in comparisons if not comparison.passed}

  assert "stop_peak_decel" in failed
  assert "final_lead_gap" in failed


def test_stopped_lead_approach_allows_valid_peak_decel_and_long_gap_from_fuzz():
  output = np.array([
    [0.0, 0.0, 90.0, 20.0, 18.0, 0.0, 90.0],
    [1.0, 18.0, 106.0, 16.0, 12.0, -1.2, 88.0],
    [2.0, 30.0, 114.0, 10.0, 4.0, -2.8, 84.0],
    [3.0, 36.0, 116.0, 3.0, 0.0, -1.0, 80.0],
    [4.0, 37.0, 116.0, 0.0, 0.0, 0.0, 11.8],
  ])

  comparisons = compare_scenario_output("stopped_lead_approach", output)
  by_metric = {comparison.metric: comparison for comparison in comparisons}

  assert by_metric["stop_peak_decel"].passed
  assert by_metric["final_lead_gap"].passed


def test_stopped_lead_jerk_uses_windowed_command_instead_of_plant_stop_snap():
  output = np.array([
    [0.00, 0.0, 5.0, 0.8, 0.0, 0.0, 5.0],
    [0.05, 0.0, 5.0, 0.5, 0.0, -0.5, 5.0],
    [0.10, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
    [0.15, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
    [0.20, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
  ])
  commanded = np.array([-0.2, -0.25, -0.3, -0.35, -0.4])

  raw = {c.metric: c for c in compare_scenario_output("stopped_lead_approach", output)}
  commanded_result = {
    c.metric: c for c in compare_scenario_output(
      "stopped_lead_approach", output, commanded_accel=commanded, jerk_window=2,
    )
  }

  assert raw["max_abs_jerk"].current == 10.0
  assert not raw["max_abs_jerk"].passed
  assert commanded_result["max_abs_jerk"].current == 1.0
  assert commanded_result["max_abs_jerk"].passed


def test_lead_approach_comparison_reports_closing_and_time_gap():
  output = np.array([
    [0.0, 0.0, 45.0, 15.0, 14.0, 0.0, 45.0],
    [1.0, 15.0, 59.0, 15.0, 14.0, -0.1, 44.0],
    [2.0, 30.0, 73.0, 14.5, 14.0, -0.2, 43.0],
    [3.0, 44.0, 87.0, 13.8, 14.0, -0.2, 43.0],
  ])

  comparisons = compare_scenario_output("slower_cut_in", output)
  by_metric = {comparison.metric: comparison for comparison in comparisons}

  assert by_metric["max_closing_speed"].current == 0.5
  assert by_metric["min_time_gap"].passed


def test_lead_approach_comparison_ignores_initial_unavoidable_closing_speed():
  output = np.array([
    [0.0, 0.0, 100.0, 22.0, 13.0, 0.0, 100.0],
    [1.0, 21.0, 113.0, 19.0, 13.0, -0.6, 92.0],
    [2.0, 38.0, 126.0, 15.0, 13.0, -0.8, 88.0],
    [3.0, 52.0, 139.0, 14.0, 13.0, -0.2, 30.0],
  ])

  comparisons = compare_scenario_output("udacity_acc_slower_lead", output)
  by_metric = {comparison.metric: comparison for comparison in comparisons}

  assert by_metric["max_closing_speed"].current == 1.0
  assert by_metric["max_closing_speed"].passed


def test_lead_approach_allows_open_gap_time_gap_to_stay_large():
  output = np.array([
    [0.0, 0.0, 80.0, 10.0, 10.0, 0.0, 80.0],
    [1.0, 10.0, 90.0, 10.0, 10.0, 0.0, 80.0],
    [2.0, 20.0, 100.0, 10.0, 10.0, 0.0, 80.0],
    [3.0, 30.0, 110.0, 10.0, 10.0, 0.0, 80.0],
  ])

  comparisons = compare_scenario_output("lead_occlusion", output)
  by_metric = {comparison.metric: comparison for comparison in comparisons}

  assert by_metric["min_time_gap"].current == 8.0
  assert by_metric["min_time_gap"].passed


def test_behavior_outline_renders_current_and_expected_behavior_table():
  outline = render_behavior_outline()

  assert "| Area | Current Behavior | Expected Behavior |" in outline
  assert "Launch from stop" in outline
  assert "Lead approach" in outline
