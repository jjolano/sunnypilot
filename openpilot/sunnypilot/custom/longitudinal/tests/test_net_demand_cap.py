from __future__ import annotations

from array import array
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import pytest

from openpilot.sunnypilot.custom.longitudinal.finalizer import CustomLongitudinalFinalizer, FinalizerResult
from openpilot.sunnypilot.custom.longitudinal.modes import LongitudinalMode
from openpilot.sunnypilot.custom.longitudinal.net_demand_cap import (
  GRAVITY,
  GradeProfile,
  NetDemandCapFinalStage,
  NetDemandEvidence,
  UphillGradeEstimator,
  fit_coast_samples,
  parse_profile,
  sanitize_mode,
)
from openpilot.sunnypilot.custom.longitudinal.wiring import CustomLongitudinalOutput


def profile(**overrides) -> GradeProfile:
  values = dict(
    pitch_zero_rad=0.04,
    grade_enter_percent=6.0,
    grade_hysteresis_percent=1.0,
    entry_dwell_s=0.10,
    exit_dwell_s=0.10,
    min_speed_mps=10.0,
    max_speed_mps=35.0,
    median_window_s=0.25,
    filter_tau_s=0.35,
    max_pose_age_s=0.25,
    max_pitch_std_rad=0.05,
    max_dynamic_a_ego=0.5,
    max_dynamic_jerk=2.0,
    max_grade_hold_s=1.0,
    calibration_rpy=(0.01, 0.02, 0.03),
    max_calibration_delta_rad=0.01,
    fit_slope=-8.0,
    fit_score=0.8,
    fit_pitch_span=0.08,
    fit_residual_mad=0.08,
    fit_sample_count=1000,
    fit_speed_band_spread=0.005,
  )
  values.update(overrides)
  return GradeProfile(**values)


def evidence(*, mode="apply", grade_percent=8.0, research=True, lead=False, p=None) -> NetDemandEvidence:
  p = p or profile()
  relative_pitch = math.atan(grade_percent / 100.0)
  return NetDemandEvidence(
    mode=mode,
    ceiling=1.2,
    profile=p,
    profile_ready=True,
    source_healthy=True,
    block_reason="",
    source_age_s=0.05,
    pitch_zero=p.pitch_zero_rad,
    relative_pitch=relative_pitch,
    filtered_grade_percent=grade_percent,
    grade_accel=GRAVITY * math.sin(relative_pitch),
    long_active=True,
    has_lead=lead,
    v_ego=20.0,
    research_actuation_allowed=research,
  )


def test_coast_fit_recovers_pitch_zero_with_outlier() -> None:
  rows = []
  for i in range(80):
    pitch = 0.01 + 0.0008 * i
    accel = -8.0 * pitch + 0.32
    rows.append((pitch, accel, 12.0 if i < 40 else 18.0))
  rows.append((0.04, 8.0, 12.0))
  result = fit_coast_samples(rows)
  assert result.ready
  assert result.pitch_zero_rad == pytest.approx(0.04, abs=1e-6)
  assert result.slope == pytest.approx(-8.0, abs=1e-6)


def test_profile_requires_explicit_calibration_marker() -> None:
  raw = {"version": 1, "calibrated": True, **profile().__dict__}
  raw["calibration_rpy"] = list(raw["calibration_rpy"])
  assert parse_profile(json.dumps(raw)) == profile()
  raw["calibrated"] = False
  assert parse_profile(json.dumps(raw)) is None
  assert sanitize_mode("bad") == "off"


def test_shadow_and_uncalibrated_apply_are_exact_pass_through() -> None:
  stage = NetDemandCapFinalStage()
  target, trace = stage.apply(0.8, evidence(mode="shadow"), should_stop=False, dt=0.05)
  assert target == 0.8
  assert trace.would_cap and not trace.applied

  uncalibrated = NetDemandEvidence(
    mode="apply", ceiling=1.2, block_reason="profile_not_calibrated",
    grade_accel=0.8, filtered_grade_percent=8.0, long_active=True,
    v_ego=20.0, research_actuation_allowed=True,
  )
  target, trace = stage.apply(0.8, uncalibrated, should_stop=False, dt=0.05)
  assert target == 0.8
  assert trace.effective_mode == "shadow"
  assert trace.block_reason == "profile_not_calibrated"

  target, trace = stage.apply(0.8, evidence(research=False), should_stop=False, dt=0.05)
  assert target == 0.8
  assert trace.effective_mode == "shadow"
  assert trace.block_reason == "research_gate_off"

  target, trace = stage.apply(
    0.8, replace(evidence(), grade_accel=math.nan), should_stop=False, dt=0.05,
  )
  assert target == 0.8
  assert not trace.applied
  assert trace.block_reason == "grade_unavailable"


def test_calibrated_apply_uses_hysteresis_and_bypasses_leads() -> None:
  stage = NetDemandCapFinalStage()
  target, trace = stage.apply(0.8, evidence(), should_stop=False, dt=0.05)
  assert target == 0.8 and trace.regime == "hold"
  target, trace = stage.apply(0.8, evidence(), should_stop=False, dt=0.05)
  expected_cap = 1.2 - evidence().grade_accel
  assert target == pytest.approx(expected_cap)
  assert trace.regime == "cap" and trace.applied

  target, trace = stage.apply(0.8, evidence(lead=True), should_stop=False, dt=0.05)
  assert target == 0.8
  assert trace.block_reason == "lead_present"
  assert trace.regime == "hold"


def test_estimator_uses_calibrated_profile_and_source_health() -> None:
  estimator = UphillGradeEstimator()
  estimator.mode = "apply"
  estimator.ceiling = 1.2
  estimator.profile = profile()
  relative_pitch = math.atan(0.08)
  for _ in range(5):
    result = estimator.update(
      car_pitch=0.04 + relative_pitch,
      live_pose_pitch=relative_pitch,
      pitch_std=0.01,
      source_age_s=0.05,
      source_valid=True,
      calibration_valid=True,
      # cereal List(Float32) is a dynamic sequence rather than list/tuple.
      calibration_rpy=array("f", [0.01, 0.02, 0.03]),
      v_ego=20.0,
      a_ego=0.0,
      long_active=True,
      gas_pressed=False,
      brake_pressed=False,
      force_decel=False,
      has_lead=False,
      research_actuation_allowed=True,
      dt=0.05,
    )
  assert result.profile_ready
  assert result.filtered_grade_percent == pytest.approx(8.0)
  assert result.grade_accel == pytest.approx(GRAVITY * math.sin(relative_pitch))

  mismatch = estimator.update(
    car_pitch=0.04 + relative_pitch,
    live_pose_pitch=relative_pitch,
    pitch_std=0.01,
    source_age_s=0.05,
    source_valid=True,
    calibration_valid=True,
    calibration_rpy=array("f", [0.01, 0.05, 0.03]),
    v_ego=20.0,
    a_ego=0.0,
    long_active=True,
    gas_pressed=False,
    brake_pressed=False,
    force_decel=False,
    has_lead=False,
    research_actuation_allowed=True,
    dt=0.05,
  )
  assert mismatch.grade_accel is None
  assert mismatch.block_reason == "calibration_mismatch"

  rewarming = estimator.update(
    car_pitch=0.04 + relative_pitch,
    live_pose_pitch=relative_pitch,
    pitch_std=0.01,
    source_age_s=0.05,
    source_valid=True,
    calibration_valid=True,
    calibration_rpy=array("f", [0.01, 0.02, 0.03]),
    v_ego=20.0,
    a_ego=0.0,
    long_active=True,
    gas_pressed=False,
    brake_pressed=False,
    force_decel=False,
    has_lead=False,
    research_actuation_allowed=True,
    dt=0.05,
  )
  assert rewarming.grade_accel is None
  assert rewarming.block_reason == "grade_stale"


def test_finalizer_records_post_cap_target() -> None:
  finalizer = CustomLongitudinalFinalizer(SimpleNamespace())
  finalizer._finalize_impl = lambda *args: FinalizerResult(0.8, False, False)  # type: ignore[method-assign]
  output = CustomLongitudinalOutput(
    a_target=0.8,
    should_stop=False,
    enabled=True,
    mode=LongitudinalMode.SCC,
    selected_intent="cruise",
    reason="cruise",
    uphill_net_demand=evidence(p=profile(entry_dwell_s=0.0)),
  )
  result = finalizer.finalize(
    {}, SimpleNamespace(enabled=True), output, False, False, 0.05, 0.8, False, 0.8, False,
    lambda *args: 0.8, lambda: None,
  )
  assert result.a_target < 0.8
  assert finalizer.final_a_prev == result.a_target
  assert result.custom_long_output_telemetry.uphill_net_demand_trace.applied
