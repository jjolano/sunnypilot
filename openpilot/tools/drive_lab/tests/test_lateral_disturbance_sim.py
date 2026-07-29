import pytest

from openpilot.tools.drive_lab.lateral_disturbance_sim import (
  LateralDisturbanceConfig,
  LateralTrace,
  lateral_disturbance_report_to_specs,
  render_lateral_disturbance_report,
  simulate_lateral_disturbance,
  synthetic_lateral_trace,
)
from openpilot.tools.drive_lab.scenario_spec import ScenarioSpec


def test_lateral_disturbance_sim_is_seed_deterministic():
  config = LateralDisturbanceConfig(seed=42, duration_s=20.0, model_noise_std=2e-5, sensor_noise_std=1e-5)
  trace = synthetic_lateral_trace(config, "sine")

  first = simulate_lateral_disturbance(trace, config)
  second = simulate_lateral_disturbance(trace, config)

  assert first.to_dict() == second.to_dict()
  assert first.config_hash == second.config_hash


def test_stiction_and_backlash_increase_fast_reversal_score():
  base_config = LateralDisturbanceConfig(seed=3, duration_s=30.0, actuator_rate_limit_deg_s=220.0)
  disturbed_config = LateralDisturbanceConfig(
    seed=3,
    duration_s=30.0,
    steering_stiction_deg=0.08,
    steering_backlash_deg=0.12,
    model_noise_std=4e-5,
    actuator_rate_limit_deg_s=220.0,
  )
  base = simulate_lateral_disturbance(synthetic_lateral_trace(base_config, "reversal"), base_config)
  disturbed = simulate_lateral_disturbance(synthetic_lateral_trace(disturbed_config, "reversal"), disturbed_config)

  assert disturbed.fast_reversal_count >= base.fast_reversal_count
  assert disturbed.steering_rate_p95 > base.steering_rate_p95


def test_authority_attenuation_reports_rebound_event():
  config = LateralDisturbanceConfig(
    duration_s=40.0,
    authority_attenuation=0.55,
    authority_start_s=8.0,
    authority_end_s=18.0,
    actuator_delay_s=0.15,
    tire_lag_s=0.35,
  )
  report = simulate_lateral_disturbance(synthetic_lateral_trace(config, "curve_entry"), config)

  assert report.rebound_count > 0
  assert any(event.kind == "rebound" for event in report.top_events)


def test_route_like_trace_reports_lag_and_render_summary():
  t = tuple(i * 0.05 for i in range(200))
  desired = tuple(0.0008 if (i // 20) % 2 == 0 else -0.0008 for i in range(200))
  trace = LateralTrace(t=t, v_ego=tuple(18.0 for _ in t), desired_curvature=desired)
  report = simulate_lateral_disturbance(trace, LateralDisturbanceConfig(tire_lag_s=0.4), source="trace")
  rendered = render_lateral_disturbance_report(report)

  assert report.sample_count == 200
  assert report.desired_actual_lag_s is not None
  assert report.desired_actual_corr is not None
  assert "Lateral disturbance simulation" in rendered
  assert report.steering_angle_pp == pytest.approx(max(desired) * 3200.0 - min(desired) * 3200.0, rel=0.05)


def test_lateral_disturbance_report_exports_route_derived_specs():
  config = LateralDisturbanceConfig(seed=7, duration_s=20.0, authority_attenuation=0.55, authority_start_s=3.0, authority_end_s=10.0)
  report = simulate_lateral_disturbance(synthetic_lateral_trace(config, "reversal"), config, source="route-a")
  specs = lateral_disturbance_report_to_specs(report, max_events=3)

  assert specs
  assert all(isinstance(spec, ScenarioSpec) for spec in specs)
  assert all(spec.mode == "lateral-disturbance" for spec in specs)
  assert specs[0].source == "route-a"
  assert specs[0].provenance["source_tool"] == "lateral_disturbance_sim"
  assert specs[0].provenance["config_hash"] == report.config_hash
