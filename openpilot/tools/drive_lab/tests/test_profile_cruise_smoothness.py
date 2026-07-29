from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openpilot.tools.drive_lab.profile_cruise_smoothness import (
  CruiseSmoothnessParams,
  analyze_route,
  load_report,
  render_report,
  save_report,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


def msg(kind: str, t_s: float, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def cruise_stream(
  times: list[float],
  accelerations: list[float],
  *,
  engaged: bool,
  system_enabled: bool | None = None,
  long_active: bool | None = None,
  include_commands: bool = True,
  command_stride: int = 1,
  speed: float = 20.0,
  speed_values: dict[int, float] | None = None,
  cruise_kph: float = 72.0,
  brake_indices: set[int] | None = None,
  gas_indices: set[int] | None = None,
  close_lead_indices: set[int] | None = None,
  stop_indices: set[int] | None = None,
  cruise_values: dict[int, float] | None = None,
  omit_channel_indices: dict[str, set[int]] | None = None,
  omit_system_enabled_indices: set[int] | None = None,
  omit_radar_indices: set[int] | None = None,
  system_enabled_values: dict[int, bool] | None = None,
  long_active_values: dict[int, bool] | None = None,
) -> list[FakeMsg]:
  brake_indices = brake_indices or set()
  gas_indices = gas_indices or set()
  close_lead_indices = close_lead_indices or set()
  stop_indices = stop_indices or set()
  cruise_values = cruise_values or {}
  speed_values = speed_values or {}
  omit_channel_indices = omit_channel_indices or {}
  omit_system_enabled_indices = omit_system_enabled_indices or set()
  omit_radar_indices = omit_radar_indices or set()
  system_enabled_values = system_enabled_values or {}
  long_active_values = long_active_values or {}
  system_enabled = engaged if system_enabled is None else system_enabled
  long_active = engaged if long_active is None else long_active
  output: list[FakeMsg] = []
  for index, (t_s, acceleration) in enumerate(zip(times, accelerations, strict=True)):
    current_speed = 0.0 if index in stop_indices else speed_values.get(index, speed)
    current_cruise = cruise_values.get(index, cruise_kph)
    if index not in omit_system_enabled_indices:
      output.append(msg("selfdriveState", t_s, enabled=system_enabled_values.get(index, system_enabled)))
    if index % command_stride == 0:
      control_fields: dict[str, Any] = {"longActive": long_active_values.get(index, long_active)}
      if include_commands:
        control_fields["actuators"] = SimpleNamespace(accel=acceleration)
      output.append(msg("carControl", t_s, **control_fields))
      if include_commands and index not in omit_channel_indices.get("carOutput", set()):
        output.append(msg("carOutput", t_s + 0.001, actuatorsOutput=SimpleNamespace(accel=acceleration)))
      if include_commands and index not in omit_channel_indices.get("longitudinalPlan", set()):
        output.append(msg("longitudinalPlan", t_s + 0.002, aTarget=acceleration))
      if include_commands and index not in omit_channel_indices.get("longitudinalPlanSP", set()):
        output.append(msg("longitudinalPlanSP", t_s + 0.003, aTarget=acceleration))
    if index not in omit_radar_indices:
      output.append(msg(
        "radarState",
        t_s + 0.004,
        leadOne=SimpleNamespace(
          present=index in close_lead_indices,
          dRel=10.0 if index in close_lead_indices else 60.0,
        ),
      ))
    output.append(msg(
      "carState",
      t_s + 0.005,
      vEgo=current_speed,
      aEgo=acceleration,
      vCruise=current_cruise,
      gasPressed=index in gas_indices,
      brakePressed=index in brake_indices,
      standstill=index in stop_indices,
    ))
  return output


def test_stable_manual_cruise_has_low_ripple_and_no_engaged_view():
  times = [i * 0.1 for i in range(201)]
  manual_report = analyze_route(
    cruise_stream(times, [0.01] * len(times), engaged=False, gas_indices=set(range(len(times)))),
    source="manual-stable",
  )
  engaged_report = analyze_route(
    cruise_stream(
      times,
      [0.01] * len(times),
      engaged=False,
      system_enabled=True,
      long_active=False,
      gas_indices=set(range(len(times))),
    ),
    source="engaged-gas-override",
  )

  assert manual_report.manual.window_count > 0
  assert manual_report.engaged.window_count == 0
  assert engaged_report.engaged.window_count == 0
  assert engaged_report.exclusion_counts["mode_unknown"] > 0
  assert engaged_report.exclusion_counts["pedal_active"] > 0
  metrics = manual_report.manual.windows[0].acceleration["a_ego"]
  assert metrics.acceleration_stddev_mps2 == pytest.approx(0.0)
  assert metrics.acceleration_peak_to_peak_mps2 == pytest.approx(0.0)
  assert metrics.acceleration_sign_reversals_per_min == pytest.approx(0.0)
  assert metrics.acceleration_deadband_share == pytest.approx(1.0)
  assert manual_report.manual.windows[0].speed_peak_to_peak_mps == pytest.approx(0.0)
  assert manual_report.manual.steady_duration_s == pytest.approx(manual_report.duration_s)
  assert "no pass/fail threshold" in render_report(manual_report)


def test_manual_gas_transition_still_excludes_nearby_samples():
  times = [i * 0.1 for i in range(301)]
  report = analyze_route(
    cruise_stream(times, [0.0] * len(times), engaged=False, gas_indices=set(range(100, 121))),
    source="manual-gas-transition",
  )

  assert report.exclusion_counts["pedal_transition"] > 0
  assert report.manual.window_count > 0
  assert all(not (window.start_s < 10.0 < window.end_s) for window in report.manual.windows)


def test_stale_mode_and_radar_are_ineligible_not_manual_no_lead():
  times = [i * 0.1 for i in range(301)]
  stale_range = set(range(80, 181))
  mode_report = analyze_route(
    cruise_stream(times, [0.0] * len(times), engaged=False, omit_system_enabled_indices=stale_range),
    source="stale-mode",
  )
  radar_report = analyze_route(
    cruise_stream(times, [0.0] * len(times), engaged=False, omit_radar_indices=stale_range),
    source="stale-radar",
  )

  assert mode_report.exclusion_counts["mode_unknown"] > 0
  assert radar_report.exclusion_counts["radar_unknown"] > 0
  assert all(not (window.start_s < 12.0 < window.end_s) for window in mode_report.manual.windows)
  assert all(not (window.start_s < 12.0 < window.end_s) for window in radar_report.manual.windows)


def test_mode_transition_bridges_unknown_samples():
  times = [i * 0.1 for i in range(401)]
  manual_indices = dict.fromkeys(range(100), False)
  report = analyze_route(
    cruise_stream(
      times,
      [0.0] * len(times),
      engaged=True,
      system_enabled=True,
      long_active=True,
      system_enabled_values=manual_indices,
      long_active_values=manual_indices,
      omit_system_enabled_indices=set(range(100, 106)),
    ),
    source="mode-unknown-bridge",
  )

  assert report.exclusion_counts["mode_unknown"] > 0
  assert report.exclusion_counts["engagement_transition"] > 0
  assert report.engaged.window_count > 0
  assert all(not (window.start_s < 10.6 < window.end_s) for window in report.engaged.windows)


def test_flickering_engaged_commands_report_jerk_and_reversals():
  times = [i * 0.1 for i in range(201)]
  accelerations = [0.3 if i % 2 == 0 else -0.3 for i in range(len(times))]
  report = analyze_route(
    cruise_stream(times, accelerations, engaged=True),
    source="engaged-flicker",
  )

  assert report.manual.window_count == 0
  assert report.engaged.window_count > 0
  metrics = report.engaged.windows[0].acceleration
  assert set(metrics) == {
    "a_ego",
    "longitudinal_plan_a_target",
    "longitudinal_plan_sp_a_target",
    "car_control_accel",
    "car_output_accel",
  }
  a_ego = metrics["a_ego"]
  assert a_ego.acceleration_stddev_mps2 == pytest.approx(0.3, abs=2e-5)
  assert a_ego.acceleration_peak_to_peak_mps2 == pytest.approx(0.6)
  assert a_ego.jerk_p90_mps3 is not None and a_ego.jerk_p90_mps3 > 5.0
  assert a_ego.acceleration_sign_reversals_per_min > 100.0
  assert a_ego.acceleration_deadband_share == pytest.approx(0.0)


def test_command_jerk_uses_source_clock_not_carstate_cadence():
  command_accelerations = [0.3 if i % 2 == 0 else -0.3 for i in range(121)]
  high_rate_times = [i * 0.01 for i in range(601)]
  high_rate_accelerations = [command_accelerations[i // 5] for i in range(len(high_rate_times))]
  low_rate_times = [i * 0.05 for i in range(121)]

  params = CruiseSmoothnessParams(window_s=4.0, step_s=2.0, min_window_s=3.0)
  high_rate = analyze_route(
    cruise_stream(high_rate_times, high_rate_accelerations, engaged=True, command_stride=5),
    source="100hz-carstate-20hz-command",
    params=params,
  )
  low_rate = analyze_route(
    cruise_stream(low_rate_times, command_accelerations, engaged=True),
    source="20hz-carstate-20hz-command",
    params=params,
  )

  high_jerk = high_rate.engaged.windows[0].acceleration["longitudinal_plan_a_target"].jerk_p90_mps3
  low_jerk = low_rate.engaged.windows[0].acceleration["longitudinal_plan_a_target"].jerk_p90_mps3
  assert high_jerk == pytest.approx(12.0)
  assert low_jerk == pytest.approx(high_jerk)


def test_transition_samples_are_excluded_from_cruise_windows():
  times = [i * 0.1 for i in range(601)]
  report = analyze_route(
    cruise_stream(
      times,
      [0.0] * len(times),
      engaged=False,
      brake_indices=set(range(50, 56)),
      close_lead_indices=set(range(300, 321)),
      stop_indices=set(range(400, 421)),
      cruise_values={100: 80.0},
    ),
    source="transitions",
  )

  assert report.exclusion_counts["pedal_active"] > 0
  assert report.exclusion_counts["pedal_transition"] > 0
  assert report.exclusion_counts["close_lead_following"] > 0
  assert report.exclusion_counts["stopping_or_launching"] > 0
  assert report.exclusion_counts["set_speed_change"] > 0
  for window in report.manual.windows:
    assert not (window.start_s < 5.0 < window.end_s)
    assert not (window.start_s < 30.0 < window.end_s)
    assert not (window.start_s < 40.0 < window.end_s)


def test_discontinuous_time_is_segmented_not_interpolated():
  times = [i * 0.1 for i in range(60)] + [12.0 + i * 0.1 for i in range(60)]
  report = analyze_route(
    cruise_stream(times, [0.0] * len(times), engaged=False),
    source="discontinuous",
    params=CruiseSmoothnessParams(window_s=4.0, step_s=2.0, min_window_s=3.0),
  )

  assert report.exclusion_counts["time_gap"] > 0
  assert report.manual.window_count > 0
  assert all(not (window.start_s < 12.0 < window.end_s) for window in report.manual.windows)


def test_unstable_speed_segment_does_not_inflate_steady_duration():
  times = [i * 0.1 for i in range(201)]
  speed_values = {i: (18.0 if i % 2 == 0 else 22.0) for i in range(len(times))}
  report = analyze_route(
    cruise_stream(times, [0.0] * len(times), engaged=False, speed_values=speed_values),
    source="unstable-speed",
  )

  assert report.manual.window_count == 0
  assert report.manual.steady_sample_count == 0
  assert report.manual.steady_duration_s == pytest.approx(0.0)
  assert report.exclusion_counts["unstable_speed_window"] > 0


def test_stale_engaged_channel_is_omitted_and_noted():
  times = [i * 0.1 for i in range(201)]
  report = analyze_route(
    cruise_stream(
      times,
      [0.0] * len(times),
      engaged=True,
      omit_channel_indices={"longitudinalPlan": set(range(50, 151))},
    ),
    source="stale-channel",
    params=CruiseSmoothnessParams(window_s=4.0, step_s=2.0, min_window_s=3.0),
  )

  stale_window = next(window for window in report.engaged.windows if 6.0 <= window.start_s <= 8.0)
  assert "longitudinal_plan_a_target" not in stale_window.acceleration
  assert "longitudinal_plan_a_target" in report.available_acceleration_channels
  assert any("stale acceleration channels" in note and "longitudinal_plan_a_target" in note for note in report.notes)
  assert "coverage longitudinal_plan_a_target" in render_report(report)


def test_partial_command_gap_does_not_bridge_reversals_or_jerk():
  times = [i * 0.1 for i in range(201)]
  accelerations = [0.3 if i < 50 or i > 100 else -0.3 for i in range(len(times))]
  report = analyze_route(
    cruise_stream(
      times,
      accelerations,
      engaged=True,
      omit_channel_indices={"longitudinalPlan": set(range(50, 101))},
    ),
    source="partial-command-gap",
    params=CruiseSmoothnessParams(window_s=20.0, step_s=10.0, min_window_s=10.0),
  )

  window = report.engaged.windows[0]
  metric = window.acceleration["longitudinal_plan_a_target"]
  coverage = window.channel_coverage["longitudinal_plan_a_target"]
  assert metric.acceleration_sign_reversals_per_min == pytest.approx(0.0)
  assert coverage.coverage_percent < 80.0
  assert coverage.distinct_source_sample_count > 2


def test_singleton_qlog_command_segments_have_nullable_reversal_rate(tmp_path):
  times = [i * 0.1 for i in range(201)]
  available_indices = {10, 150}
  report = analyze_route(
    cruise_stream(
      times,
      [0.0] * len(times),
      engaged=True,
      omit_channel_indices={"longitudinalPlan": set(range(len(times))) - available_indices},
    ),
    source="singleton-qlog-command",
    params=CruiseSmoothnessParams(window_s=20.0, step_s=10.0, min_window_s=10.0),
  )

  metric = report.engaged.windows[0].acceleration["longitudinal_plan_a_target"]
  assert metric.duration_s == pytest.approx(0.0)
  assert metric.acceleration_sign_reversals_per_min is None
  path = tmp_path / "smoothness.json"
  save_report(report, path)
  loaded = load_report(path)
  loaded_metric = loaded.engaged.windows[0].acceleration["longitudinal_plan_a_target"]
  assert loaded_metric.acceleration_sign_reversals_per_min is None
  assert "reversals_per_min=n/a" in render_report(loaded)


def test_route_reports_all_accepted_windows_without_detail_cap():
  times = [i * 0.1 for i in range(601)]
  report = analyze_route(
    cruise_stream(times, [0.0] * len(times), engaged=False),
    source="more-than-fifty-windows",
    params=CruiseSmoothnessParams(window_s=5.0, step_s=1.0, min_window_s=5.0),
  )

  assert report.manual.window_count > 50
  assert report.manual.window_count == len(report.manual.windows)
  assert report.manual.steady_duration_s == pytest.approx(report.duration_s)


def test_missing_engaged_acceleration_channels_are_explicitly_omitted():
  times = [i * 0.1 for i in range(101)]
  report = analyze_route(
    cruise_stream(times, [0.0] * len(times), engaged=True, include_commands=False),
    source="missing-channels",
  )

  assert report.available_acceleration_channels == ["a_ego"]
  assert report.engaged.window_count > 0
  assert set(report.engaged.windows[0].acceleration) == {"a_ego"}
  assert any("car_control_accel" in note for note in report.notes)
