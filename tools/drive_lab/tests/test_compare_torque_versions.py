from openpilot.tools.drive_lab.compare_torque_versions import (
  TorqueVersionMetrics,
  acceptance_check,
  compare_torque_versions,
  render_compare_torque_versions,
)


def test_compare_torque_versions_returns_metrics_per_version():
  metrics = {
    "2.1": TorqueVersionMetrics(
      version="2.1",
      turn_in_80pct_time_s=0.4,
      turn_in_overshoot=0.1,
      steady_curve_mean_abs_error=0.05,
      turn_exit_zero_time_s=0.8,
      straight_road_torque_sign_flips=2,
    ),
    "4.1": TorqueVersionMetrics(
      version="4.1",
      turn_in_80pct_time_s=0.5,
      steady_curve_mean_abs_error=0.06,
      turn_exit_zero_time_s=0.9,
      straight_road_torque_sign_flips=2,
    ),
    "5.0": TorqueVersionMetrics(
      version="5.0",
      turn_in_80pct_time_s=0.4,
      steady_curve_mean_abs_error=0.05,
      turn_exit_zero_time_s=0.7,
      straight_road_torque_sign_flips=1,
      v5_active_pct=72.0,
      preview_boost_applied_max=0.05,
    ),
  }
  report = compare_torque_versions(metrics, scenario="clean_turn_in")
  assert report.scenario == "clean_turn_in"
  assert [m.version for m in report.metrics] == ["2.1", "4.1", "5.0"]


def test_compare_torque_versions_acceptance_passes_when_v5_meets_criteria():
  metrics = {
    "2.1": TorqueVersionMetrics(
      version="2.1",
      turn_in_80pct_time_s=0.4,
      turn_in_overshoot=0.1,
      steady_curve_mean_abs_error=0.05,
      turn_exit_zero_time_s=0.8,
      straight_road_torque_sign_flips=2,
    ),
    "5.0": TorqueVersionMetrics(
      version="5.0",
      turn_in_80pct_time_s=0.3,
      steady_curve_mean_abs_error=0.04,
      turn_exit_zero_time_s=0.7,
      straight_road_torque_sign_flips=1,
      v5_active_pct=70.0,
    ),
  }
  report = compare_torque_versions(metrics, scenario="clean_turn_in")
  acceptance = acceptance_check(report)
  # v5 meets every criterion; the report must say so.
  assert acceptance["turn_in_80pct"] is True
  assert acceptance["turn_exit_zero"] is True
  assert acceptance["steady_curve"] is True
  assert acceptance["straight_road_flips"] is True


def test_compare_torque_versions_acceptance_fails_when_v5_lags():
  metrics = {
    "2.1": TorqueVersionMetrics(
      version="2.1",
      turn_in_80pct_time_s=0.4,
      steady_curve_mean_abs_error=0.05,
      turn_exit_zero_time_s=0.8,
      straight_road_torque_sign_flips=2,
    ),
    "5.0": TorqueVersionMetrics(
      version="5.0",
      turn_in_80pct_time_s=0.6,
      steady_curve_mean_abs_error=0.05,
      turn_exit_zero_time_s=1.0,
      straight_road_torque_sign_flips=5,
      v5_active_pct=20.0,
    ),
  }
  report = compare_torque_versions(metrics, scenario="clean_turn_in")
  acceptance = acceptance_check(report)
  assert acceptance["turn_in_80pct"] is False
  assert acceptance["turn_exit_zero"] is False
  assert acceptance["straight_road_flips"] is False


def test_compare_torque_versions_render_is_table_shaped():
  metrics = {
    "2.1": TorqueVersionMetrics(version="2.1", turn_in_80pct_time_s=0.4),
    "4.1": TorqueVersionMetrics(version="4.1", turn_in_80pct_time_s=0.5),
    "5.0": TorqueVersionMetrics(version="5.0", turn_in_80pct_time_s=0.4),
  }
  report = compare_torque_versions(metrics, scenario="clean_turn_in")
  rendered = render_compare_torque_versions(report)
  # Header + three rows
  lines = rendered.split("\n")
  assert len(lines) >= 4
  # All three version labels appear
  assert "2.1" in rendered
  assert "4.1" in rendered
  assert "5.0" in rendered
  # The scenario label appears
  assert "clean_turn_in" in rendered


def test_compare_torque_versions_to_dict_round_trips():
  metrics = {
    "5.0": TorqueVersionMetrics(
      version="5.0",
      v5_active_pct=80.0,
      preview_boost_applied_max=0.05,
      v5_disabled_reasons={"low_path_quality": 3, "wobble_active": 1},
    ),
  }
  report = compare_torque_versions(metrics, scenario="wobble")
  as_dict = report.to_dict()
  assert as_dict["scenario"] == "wobble"
  m = as_dict["metrics"][0]
  assert m["version"] == "5.0"
  assert m["v5_active_pct"] == 80.0
  assert m["v5_disabled_reasons"]["low_path_quality"] == 3
