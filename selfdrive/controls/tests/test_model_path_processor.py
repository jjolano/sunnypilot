import math
from dataclasses import replace

import pytest

from openpilot.selfdrive.controls.lib.model_path_processor import (
  LOW_QUALITY_BLEND_MIN_ALPHA,
  LOW_QUALITY_BLEND_THRESHOLD,
  LOW_SPEED_UNTRUSTED_CURVATURE_STEP,
  ModelPathProcessor,
  ModelPathProcessorInputs,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


BASE_INPUTS = ModelPathProcessorInputs(
  lat_active=True,
  v_ego=20.0,
  desired_curvature=0.002,
  measured_curvature=0.0005,
  previous_desired_curvature=0.001,
  position_x=tuple(range(ModelConstants.IDX_N)),
  position_y=tuple(0.02 * i for i in range(ModelConstants.IDX_N)),
  position_y_std=tuple(0.1 for _ in range(ModelConstants.IDX_N)),
  orientation_z=tuple(0.0 for _ in range(ModelConstants.IDX_N)),
  orientation_rate_z=tuple(0.0 for _ in range(ModelConstants.IDX_N)),
  lane_line_probs=(0.0, 0.9, 0.9, 0.0),
  turn_curvature_sign=0,
  frame_drop_perc=0.0,
)


def make_inputs(**overrides: object) -> ModelPathProcessorInputs:
  return replace(BASE_INPUTS, **overrides)


def constant_curvature_yaws(curvature: float, v_ego: float) -> tuple[float, ...]:
  return tuple(float(v_ego * curvature * t) for t in ModelConstants.T_IDXS)


def constant_curvature_yaw_rates(curvature: float, v_ego: float) -> tuple[float, ...]:
  return tuple(float(v_ego * curvature) for _ in ModelConstants.T_IDXS)


def test_good_path_passes_curvature_unchanged():
  result = ModelPathProcessor().update(make_inputs())

  assert not result.gated
  assert result.reason == "ok"
  assert result.quality == pytest.approx(1.0)
  assert result.desired_curvature == pytest.approx(0.002)


def test_smoothed_model_path_curvature_toggle_off_preserves_raw_curvature():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.003,
    orientation_z=constant_curvature_yaws(0.0, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.0, 20.0),
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(0.003)


def test_smoothed_model_path_curvature_matches_constant_curvature_path():
  curvature = 0.002
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=curvature,
    previous_desired_curvature=curvature,
    orientation_z=constant_curvature_yaws(curvature, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(curvature, 20.0),
    smooth_model_path_curvature=True,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(curvature)


def test_smoothed_model_path_curvature_blends_toward_local_fit():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.003,
    previous_desired_curvature=0.002,
    orientation_z=constant_curvature_yaws(0.0, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.0, 20.0),
    smooth_model_path_curvature=True,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert 0.002 <= result.desired_curvature < 0.003


def test_smoothed_model_path_curvature_disables_at_low_speed():
  result = ModelPathProcessor().update(make_inputs(
    v_ego=3.0,
    desired_curvature=0.003,
    orientation_z=constant_curvature_yaws(0.0, 3.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.0, 3.0),
    smooth_model_path_curvature=True,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(0.003)


def test_smoothed_model_path_curvature_lane_change_starts_with_full_fade_and_smoothed_handover():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.003,
    orientation_z=constant_curvature_yaws(0.0, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.0, 20.0),
    smooth_model_path_curvature=True,
    lane_change_active=True,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.lane_change_fade == pytest.approx(1.0)
  assert 0.002 <= result.desired_curvature <= 0.003


def test_lane_change_fade_decays_across_frames():
  processor = ModelPathProcessor()
  base = dict(
    desired_curvature=0.003,
    orientation_z=constant_curvature_yaws(0.0, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.0, 20.0),
    smooth_model_path_curvature=True,
    lane_change_active=True,
  )
  first = processor.update(make_inputs(**base))
  assert first.lane_change_fade == pytest.approx(1.0)
  second = processor.update(make_inputs(previous_desired_curvature=first.desired_curvature, **base))
  assert second.lane_change_fade < first.lane_change_fade


def test_smoothed_path_publishes_nonzero_damping_telemetry_when_active():
  curvature = 0.002
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=curvature,
    previous_desired_curvature=curvature,
    orientation_z=constant_curvature_yaws(curvature, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(curvature, 20.0),
    smooth_model_path_curvature=True,
  ))
  assert result.damping_alpha > 0.0
  assert result.smoothing_tau_s > 0.0
  assert result.lane_change_fade == pytest.approx(0.0)


def test_frame_drop_bumps_trust_penalty_then_decays():
  processor = ModelPathProcessor()
  gated = processor.update(make_inputs(frame_drop_perc=50.0))
  assert gated.trust_penalty > 0.25
  trust = gated.trust_penalty
  for _ in range(40):
    gated = processor.update(make_inputs(previous_desired_curvature=gated.desired_curvature, frame_drop_perc=0.0))
  assert gated.trust_penalty < trust * 0.5


def test_near_zero_desired_curvature_softens_spatial_correction_vs_mid_curvature():
  flat = dict(
    orientation_z=constant_curvature_yaws(0.0, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.0, 20.0),
    smooth_model_path_curvature=True,
  )
  hi_d, lo_d = 0.003, 0.00008
  hi = ModelPathProcessor().update(make_inputs(desired_curvature=hi_d, previous_desired_curvature=0.002, **flat))
  lo = ModelPathProcessor().update(make_inputs(desired_curvature=lo_d, previous_desired_curvature=0.00007, **flat))
  hi_rel = abs(hi.spatial_smoothed_curvature - hi_d) / max(abs(hi_d), 1e-9)
  lo_rel = abs(lo.spatial_smoothed_curvature - lo_d) / max(abs(lo_d), 1e-9)
  assert lo_rel < hi_rel


def test_temporal_damping_limits_single_frame_spike_in_smoothed_output():
  processor = ModelPathProcessor()
  stable = dict(
    v_ego=20.0,
    orientation_z=constant_curvature_yaws(0.002, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.002, 20.0),
    smooth_model_path_curvature=True,
  )
  processor.update(make_inputs(desired_curvature=0.002, previous_desired_curvature=0.002, **stable))
  spike = processor.update(make_inputs(desired_curvature=0.006, previous_desired_curvature=0.002, **stable))
  assert spike.desired_curvature < 0.006
  assert spike.desired_curvature > 0.002


def test_smoothed_model_path_curvature_rejects_large_raw_disagreement():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.006,
    previous_desired_curvature=0.005,
    orientation_z=constant_curvature_yaws(0.0, 20.0),
    orientation_rate_z=constant_curvature_yaw_rates(0.0, 20.0),
    smooth_model_path_curvature=True,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(0.006)


def test_inactive_returns_measured_curvature():
  result = ModelPathProcessor().update(make_inputs(lat_active=False))

  assert result.gated
  assert result.reason == "inactive"
  assert result.desired_curvature == pytest.approx(0.0005)


def test_inactive_does_not_compute_unused_fallbacks(monkeypatch):
  monkeypatch.setattr(ModelPathProcessor, "_fallback_curvature", staticmethod(lambda *_: pytest.fail("soft fallback should not be computed")))
  monkeypatch.setattr(ModelPathProcessor, "_hard_invalid_fallback_curvature", classmethod(lambda *_: pytest.fail("hard fallback should not be computed")))

  result = ModelPathProcessor().update(make_inputs(lat_active=False))

  assert result.gated
  assert result.reason == "inactive"
  assert result.desired_curvature == pytest.approx(0.0005)


def test_nonfinite_curvature_decays_toward_measured_curvature():
  result = ModelPathProcessor().update(make_inputs(desired_curvature=math.nan))

  assert result.gated
  assert result.reason == "nonfinite_curvature"
  assert 0.0005 < result.desired_curvature < 0.001


def test_nonfinite_curvature_does_not_compute_soft_fallback(monkeypatch):
  monkeypatch.setattr(ModelPathProcessor, "_fallback_curvature", staticmethod(lambda *_: pytest.fail("soft fallback should not be computed")))

  result = ModelPathProcessor().update(make_inputs(desired_curvature=math.nan))

  assert result.gated
  assert result.reason == "nonfinite_curvature"
  assert 0.0005 < result.desired_curvature < 0.001


def test_good_path_does_not_compute_hard_invalid_fallback(monkeypatch):
  monkeypatch.setattr(ModelPathProcessor, "_hard_invalid_fallback_curvature", classmethod(lambda *_: pytest.fail("hard fallback should not be computed")))

  result = ModelPathProcessor().update(make_inputs())

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(0.002)


def test_missing_path_decays_toward_measured_curvature():
  result = ModelPathProcessor().update(make_inputs(position_x=(0.0, 1.0), position_y=(0.0, 0.1)))

  assert result.gated
  assert result.reason == "invalid_path"
  assert 0.0005 < result.desired_curvature < 0.001


def test_repeated_invalid_path_frames_continue_toward_measured_curvature():
  processor = ModelPathProcessor()
  first = processor.update(make_inputs(position_x=(0.0, 1.0), position_y=(0.0, 0.1), previous_desired_curvature=0.004, measured_curvature=0.001))
  second = processor.update(make_inputs(position_x=(0.0, 1.0), position_y=(0.0, 0.1), previous_desired_curvature=first.desired_curvature, measured_curvature=0.001))

  assert abs(second.desired_curvature - 0.001) < abs(first.desired_curvature - 0.001)


def test_invalid_path_recovery_limits_high_lat_curve_exit_release():
  processor = ModelPathProcessor()
  invalid = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0050,
    measured_curvature=-0.0118,
    previous_desired_curvature=-0.0118,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))

  recovered = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0042,
    measured_curvature=-0.0117,
    previous_desired_curvature=invalid.desired_curvature,
  ))

  assert invalid.gated
  assert invalid.reason == "invalid_path"
  assert recovered.gated
  assert recovered.reason == "invalid_path"
  assert recovered.desired_curvature < -0.0110


def test_invalid_path_recovery_allows_higher_same_direction_curvature_demand():
  processor = ModelPathProcessor()
  invalid = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0050,
    measured_curvature=-0.0118,
    previous_desired_curvature=-0.0118,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))

  recovered = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0120,
    measured_curvature=-0.0117,
    previous_desired_curvature=invalid.desired_curvature,
  ))

  assert not recovered.gated
  assert recovered.reason == "ok"
  assert recovered.desired_curvature == pytest.approx(-0.0120)


def test_invalid_path_recovery_still_limits_after_brief_higher_same_direction_demand():
  processor = ModelPathProcessor()
  invalid = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0050,
    measured_curvature=-0.0118,
    previous_desired_curvature=-0.0118,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))
  higher_demand = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0120,
    measured_curvature=-0.0117,
    previous_desired_curvature=invalid.desired_curvature,
  ))

  recovered = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0042,
    measured_curvature=-0.0117,
    previous_desired_curvature=higher_demand.desired_curvature,
  ))

  assert not higher_demand.gated
  assert higher_demand.desired_curvature == pytest.approx(-0.0120)
  assert recovered.gated
  assert recovered.reason == "invalid_path"
  assert recovered.desired_curvature < -0.0110


def test_invalid_path_recovery_state_clears_after_soft_gate():
  processor = ModelPathProcessor()
  invalid = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0050,
    measured_curvature=-0.0118,
    previous_desired_curvature=-0.0118,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))
  soft_gate = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0042,
    measured_curvature=-0.0117,
    previous_desired_curvature=invalid.desired_curvature,
    frame_drop_perc=50.0,
  ))

  recovered = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0120,
    measured_curvature=-0.0117,
    previous_desired_curvature=soft_gate.desired_curvature,
  ))

  assert soft_gate.gated
  assert soft_gate.reason == "frame_drop"
  assert recovered.gated
  assert recovered.reason == "frame_drop"


def test_invalid_path_recovery_state_clears_after_different_gate():
  processor = ModelPathProcessor()
  invalid = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0050,
    measured_curvature=-0.0118,
    previous_desired_curvature=-0.0118,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))
  opposite = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=0.0040,
    measured_curvature=0.0,
    previous_desired_curvature=invalid.desired_curvature,
    turn_curvature_sign=-1,
  ))

  recovered = processor.update(make_inputs(
    v_ego=15.5,
    desired_curvature=-0.0120,
    measured_curvature=0.0,
    previous_desired_curvature=opposite.desired_curvature,
  ))

  assert opposite.gated
  assert opposite.reason == "turn_opposite_curvature"
  assert not recovered.gated
  assert recovered.reason == "ok"
  assert recovered.desired_curvature == pytest.approx(-0.0120)


def test_degenerate_finite_path_decays_toward_measured_curvature():
  result = ModelPathProcessor().update(make_inputs(position_x=tuple(0.0 for _ in range(ModelConstants.IDX_N))))

  assert result.gated
  assert result.reason == "invalid_path"
  assert 0.0005 < result.desired_curvature < 0.001


def test_far_horizon_foldback_with_valid_core_path_passes_curvature_unchanged():
  position_x = list(range(ModelConstants.IDX_N))
  position_x[-1] = position_x[-2] - 1

  result = ModelPathProcessor().update(make_inputs(position_x=tuple(position_x)))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(0.002)


def test_nonfinite_path_sample_decays_toward_measured_curvature():
  position_y = [0.0 for _ in range(ModelConstants.IDX_N)]
  position_y[5] = math.inf

  result = ModelPathProcessor().update(make_inputs(position_y=tuple(position_y)))

  assert result.gated
  assert result.reason == "invalid_path"
  assert 0.0005 < result.desired_curvature < 0.001


def test_core_path_with_lateral_discontinuity_is_invalid():
  position_y = [0.02 * i for i in range(ModelConstants.IDX_N)]
  position_y[6] += 3.0

  result = ModelPathProcessor().update(make_inputs(position_y=tuple(position_y)))

  assert result.gated
  assert result.reason == "invalid_path"
  assert 0.0005 < result.desired_curvature < 0.001


def test_curved_core_path_with_large_lateral_step_over_long_x_distance_passes():
  position_x = [float(i) for i in range(ModelConstants.IDX_N)]
  position_y = [0.04 * i for i in range(ModelConstants.IDX_N)]
  position_x[16] = position_x[15] + 6.0
  position_y[16] = position_y[15] + 1.8

  result = ModelPathProcessor().update(make_inputs(
    v_ego=21.0,
    desired_curvature=0.006,
    measured_curvature=0.004,
    previous_desired_curvature=0.004,
    position_x=tuple(position_x),
    position_y=tuple(position_y),
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(0.006)


def test_implausible_curvature_jump_is_held():
  result = ModelPathProcessor().update(make_inputs(desired_curvature=0.03, previous_desired_curvature=0.0))

  assert result.gated
  assert result.reason == "curvature_jump"
  assert result.desired_curvature == pytest.approx(0.0)


def test_low_lane_confidence_degrades_quality_without_hard_gate():
  result = ModelPathProcessor().update(make_inputs(lane_line_probs=(0.0, 0.1, 0.2, 0.0)))

  assert not result.gated
  assert result.reason == "low_lane_confidence"
  assert 0.0 < result.quality < 1.0
  assert result.desired_curvature == pytest.approx(0.002)


def test_sustained_low_lane_confidence_eventually_blends_but_single_frame_does_not():
  processor = ModelPathProcessor()
  low_lane_inputs = dict(
    v_ego=3.0,
    desired_curvature=0.006,
    measured_curvature=0.001,
    previous_desired_curvature=0.001,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
  )

  first = processor.update(make_inputs(**low_lane_inputs))
  second = processor.update(make_inputs(**low_lane_inputs))
  sustained = processor.update(make_inputs(**low_lane_inputs))

  assert not first.gated
  assert first.reason == "low_lane_confidence"
  assert first.quality > LOW_QUALITY_BLEND_THRESHOLD
  assert first.desired_curvature == pytest.approx(0.006)
  assert not second.gated
  assert sustained.gated
  assert sustained.reason == "low_lane_confidence"
  assert sustained.quality < LOW_QUALITY_BLEND_THRESHOLD
  assert 0.001 < sustained.desired_curvature < 0.006


def test_high_speed_sustained_low_lane_confidence_does_not_soft_gate():
  processor = ModelPathProcessor()
  low_lane_inputs = dict(
    v_ego=20.0,
    desired_curvature=0.006,
    measured_curvature=0.001,
    previous_desired_curvature=0.001,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
  )

  first = processor.update(make_inputs(**low_lane_inputs))
  second = processor.update(make_inputs(**low_lane_inputs))
  third = processor.update(make_inputs(**low_lane_inputs))

  assert not first.gated
  assert not second.gated
  assert not third.gated
  assert third.reason == "low_lane_confidence"
  assert third.quality > LOW_QUALITY_BLEND_THRESHOLD
  assert third.desired_curvature == pytest.approx(0.006)


def test_invalid_path_breaks_sustained_low_lane_confidence_streak():
  processor = ModelPathProcessor()
  low_lane_inputs = dict(
    v_ego=3.0,
    desired_curvature=0.006,
    measured_curvature=0.001,
    previous_desired_curvature=0.001,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
  )

  first = processor.update(make_inputs(**low_lane_inputs))
  second = processor.update(make_inputs(**low_lane_inputs))
  invalid = processor.update(make_inputs(
    **low_lane_inputs,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))
  after_invalid_inputs = {**low_lane_inputs, "previous_desired_curvature": invalid.desired_curvature}
  after_invalid = processor.update(make_inputs(**after_invalid_inputs))

  assert not first.gated
  assert not second.gated
  assert invalid.gated
  assert invalid.reason == "invalid_path"
  assert not after_invalid.gated
  assert after_invalid.reason == "low_lane_confidence"
  assert after_invalid.quality > LOW_QUALITY_BLEND_THRESHOLD


def test_low_speed_low_lane_confidence_uses_retained_plausible_curve():
  processor = ModelPathProcessor()
  seed = processor.update(make_inputs(
    v_ego=6.0,
    desired_curvature=0.020,
    measured_curvature=0.018,
    previous_desired_curvature=0.018,
  ))
  low_lane_inputs = dict(
    v_ego=6.0,
    desired_curvature=0.020,
    measured_curvature=0.018,
    previous_desired_curvature=0.001,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
  )

  processor.update(make_inputs(**low_lane_inputs))
  processor.update(make_inputs(**low_lane_inputs))
  retained = processor.update(make_inputs(**low_lane_inputs))

  assert seed.reason == "ok"
  assert retained.gated
  assert retained.reason == "low_lane_confidence"
  assert retained.desired_curvature == pytest.approx(0.020)


def test_retained_curve_rejects_opposite_sign_model_curvature():
  processor = ModelPathProcessor()
  processor.update(make_inputs(
    v_ego=6.0,
    desired_curvature=0.020,
    measured_curvature=0.018,
    previous_desired_curvature=0.018,
  ))
  low_lane_inputs = dict(
    v_ego=6.0,
    desired_curvature=-0.020,
    measured_curvature=0.018,
    previous_desired_curvature=0.001,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
  )

  processor.update(make_inputs(**low_lane_inputs))
  processor.update(make_inputs(**low_lane_inputs))
  rejected = processor.update(make_inputs(**low_lane_inputs))

  assert rejected.gated
  assert rejected.reason == "low_lane_confidence"
  assert rejected.desired_curvature == pytest.approx(0.001 - LOW_SPEED_UNTRUSTED_CURVATURE_STEP)


def test_low_speed_invalid_path_keeps_compatible_retained_curve():
  processor = ModelPathProcessor()
  seed = processor.update(make_inputs(
    v_ego=6.0,
    desired_curvature=0.020,
    measured_curvature=0.018,
    previous_desired_curvature=0.018,
  ))
  invalid = processor.update(make_inputs(
    v_ego=6.0,
    desired_curvature=0.020,
    measured_curvature=0.005,
    previous_desired_curvature=seed.desired_curvature,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))

  assert seed.reason == "ok"
  assert invalid.gated
  assert invalid.reason == "invalid_path"
  assert invalid.desired_curvature == pytest.approx(seed.desired_curvature)


def test_low_speed_invalid_path_after_low_lane_confidence_limits_curvature_step():
  processor = ModelPathProcessor()
  low_lane_inputs = dict(
    v_ego=3.0,
    desired_curvature=0.020,
    measured_curvature=0.001,
    previous_desired_curvature=0.001,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
  )

  first = processor.update(make_inputs(**low_lane_inputs))
  second = processor.update(make_inputs(**{**low_lane_inputs, "previous_desired_curvature": first.desired_curvature}))
  low_lane = processor.update(make_inputs(**{**low_lane_inputs, "previous_desired_curvature": second.desired_curvature}))
  invalid = processor.update(make_inputs(
    v_ego=3.0,
    desired_curvature=0.020,
    measured_curvature=-0.080,
    previous_desired_curvature=low_lane.desired_curvature,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))

  assert low_lane.gated
  assert low_lane.reason == "low_lane_confidence"
  assert invalid.gated
  assert invalid.reason == "invalid_path"
  assert invalid.desired_curvature == pytest.approx(low_lane.desired_curvature - LOW_SPEED_UNTRUSTED_CURVATURE_STEP)


def test_retained_curve_rejects_large_magnitude_disagreement():
  processor = ModelPathProcessor()
  seed = processor.update(make_inputs(
    v_ego=11.9,
    desired_curvature=0.030,
    measured_curvature=0.028,
    previous_desired_curvature=0.028,
    orientation_z=constant_curvature_yaws(0.030, 11.9),
    orientation_rate_z=constant_curvature_yaw_rates(0.030, 11.9),
  ))
  low_lane_inputs = dict(
    v_ego=11.9,
    desired_curvature=0.001,
    measured_curvature=0.001,
    previous_desired_curvature=0.001,
    lane_line_probs=(0.0, 0.1, 0.2, 0.0),
    orientation_z=constant_curvature_yaws(0.001, 11.9),
    orientation_rate_z=constant_curvature_yaw_rates(0.001, 11.9),
  )

  first = processor.update(make_inputs(**low_lane_inputs))
  second = processor.update(make_inputs(**low_lane_inputs))
  retained_rejected = processor.update(make_inputs(**low_lane_inputs))

  assert seed.reason == "ok"
  assert not first.gated
  assert not second.gated
  assert retained_rejected.gated
  assert retained_rejected.reason == "low_lane_confidence"
  assert retained_rejected.desired_curvature == pytest.approx(0.001)


@pytest.mark.parametrize(
  ("desired_curvature", "turn_curvature_sign", "expected_curvature"),
  [
    (-0.004, 1, 0.0015),
    (0.004, -1, 0.0),
  ],
)
def test_turn_intent_suppresses_opposite_sign_curvature(desired_curvature, turn_curvature_sign, expected_curvature):
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=desired_curvature,
    previous_desired_curvature=0.0015,
    turn_curvature_sign=turn_curvature_sign,
  ))

  assert result.gated
  assert result.reason == "turn_opposite_curvature"
  assert result.desired_curvature == pytest.approx(expected_curvature)


def test_turn_intent_uses_measured_when_previous_fallback_conflicts_with_turn():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.004,
    measured_curvature=-0.0004,
    previous_desired_curvature=0.0015,
    turn_curvature_sign=-1,
  ))

  assert result.gated
  assert result.reason == "turn_opposite_curvature"
  assert result.desired_curvature == pytest.approx(-0.0004)


@pytest.mark.parametrize(
  ("desired_curvature", "turn_curvature_sign"),
  [
    (0.004, 1),
    (-0.004, -1),
  ],
)
def test_turn_intent_allows_same_sign_curvature(desired_curvature, turn_curvature_sign):
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=desired_curvature,
    turn_curvature_sign=turn_curvature_sign,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(desired_curvature)


def test_no_turn_intent_allows_opposite_direction_curvature():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=-0.004,
    turn_curvature_sign=0,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.desired_curvature == pytest.approx(-0.004)


def test_high_path_std_blends_toward_previous_desired():
  result = ModelPathProcessor().update(make_inputs(position_y_std=tuple(1.4 for _ in range(ModelConstants.IDX_N))))

  assert result.gated
  assert result.reason == "high_path_std"
  assert 0.001 < result.desired_curvature < 0.002


def test_low_speed_invalid_path_after_high_path_std_limits_curvature_step():
  processor = ModelPathProcessor()
  high_std = processor.update(make_inputs(
    v_ego=3.0,
    desired_curvature=0.020,
    measured_curvature=0.040,
    previous_desired_curvature=0.040,
    position_y_std=tuple(1.4 for _ in range(ModelConstants.IDX_N)),
  ))
  invalid = processor.update(make_inputs(
    v_ego=3.0,
    desired_curvature=0.020,
    measured_curvature=-0.080,
    previous_desired_curvature=high_std.desired_curvature,
    position_x=(0.0, 1.0),
    position_y=(0.0, 0.1),
  ))

  assert high_std.gated
  assert high_std.reason == "high_path_std"
  assert invalid.gated
  assert invalid.reason == "invalid_path"
  assert invalid.desired_curvature == pytest.approx(high_std.desired_curvature - LOW_SPEED_UNTRUSTED_CURVATURE_STEP)


def test_low_speed_high_path_std_hold_outlasts_high_speed_hold():
  position_y_std = tuple(1.4 for _ in range(ModelConstants.IDX_N))
  high_speed_processor = ModelPathProcessor()
  low_speed_processor = ModelPathProcessor()

  high_speed_gated = high_speed_processor.update(make_inputs(v_ego=20.0, position_y_std=position_y_std))
  high_speed_held_once = high_speed_processor.update(make_inputs(previous_desired_curvature=high_speed_gated.desired_curvature))
  high_speed_held_twice = high_speed_processor.update(make_inputs(previous_desired_curvature=high_speed_held_once.desired_curvature))
  high_speed_recovered = high_speed_processor.update(make_inputs(previous_desired_curvature=high_speed_held_twice.desired_curvature))

  low_speed_gated = low_speed_processor.update(make_inputs(v_ego=3.0, position_y_std=position_y_std))
  low_speed_held_once = low_speed_processor.update(make_inputs(v_ego=3.0, previous_desired_curvature=low_speed_gated.desired_curvature))
  low_speed_held_twice = low_speed_processor.update(make_inputs(v_ego=3.0, previous_desired_curvature=low_speed_held_once.desired_curvature))

  assert high_speed_gated.hold_frames_remaining == 2
  assert low_speed_gated.hold_frames_remaining > high_speed_gated.hold_frames_remaining
  assert not high_speed_recovered.gated
  assert low_speed_held_twice.gated
  assert low_speed_held_twice.reason == "high_path_std"


def test_low_speed_low_quality_blend_limits_curvature_step_from_raw_target():
  desired_curvature = 0.02
  fallback_curvature = 0.0
  result = ModelPathProcessor().update(make_inputs(
    v_ego=3.0,
    desired_curvature=desired_curvature,
    measured_curvature=fallback_curvature,
    previous_desired_curvature=fallback_curvature,
    position_y_std=tuple(1.4 for _ in range(ModelConstants.IDX_N)),
  ))

  raw_blend_alpha = LOW_QUALITY_BLEND_MIN_ALPHA + (1.0 - LOW_QUALITY_BLEND_MIN_ALPHA) * result.quality / LOW_QUALITY_BLEND_THRESHOLD
  raw_blended_target = fallback_curvature + raw_blend_alpha * (desired_curvature - fallback_curvature)

  assert result.gated
  assert result.reason == "high_path_std"
  assert raw_blended_target > LOW_SPEED_UNTRUSTED_CURVATURE_STEP
  assert result.desired_curvature == pytest.approx(LOW_SPEED_UNTRUSTED_CURVATURE_STEP)
  assert result.desired_curvature < raw_blended_target


def test_turn_intent_relaxes_same_sign_high_path_std():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.004,
    orientation_z=tuple(0.01 for _ in range(ModelConstants.IDX_N)),
    position_y_std=tuple(1.4 for _ in range(ModelConstants.IDX_N)),
    turn_curvature_sign=1,
  ))

  assert not result.gated
  assert result.reason == "ok"
  assert result.quality == pytest.approx(1.0)
  assert result.desired_curvature == pytest.approx(0.004)


def test_turn_intent_suppresses_opposite_sign_before_path_std_blending():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.004,
    orientation_z=tuple(0.01 for _ in range(ModelConstants.IDX_N)),
    position_y_std=tuple(1.4 for _ in range(ModelConstants.IDX_N)),
    turn_curvature_sign=-1,
  ))

  assert result.gated
  assert result.reason == "turn_opposite_curvature"
  assert result.desired_curvature == pytest.approx(0.0)


def test_turn_intent_disagreement_does_not_relax_high_path_std():
  result = ModelPathProcessor().update(make_inputs(
    desired_curvature=0.004,
    position_y_std=tuple(1.4 for _ in range(ModelConstants.IDX_N)),
    turn_curvature_sign=1,
  ))

  assert result.gated
  assert result.reason == "high_path_std"
  assert 0.001 < result.desired_curvature < 0.004


def test_path_curvature_disagreement_blends_toward_previous_desired():
  orientation_z = tuple(0.05 for _ in range(ModelConstants.IDX_N))
  result = ModelPathProcessor().update(make_inputs(orientation_z=orientation_z))

  assert result.gated
  assert result.reason == "path_disagreement"
  assert 0.001 < result.desired_curvature < 0.002


def test_path_disagreement_hold_prevents_single_frame_flapping():
  processor = ModelPathProcessor()
  orientation_z = tuple(0.05 for _ in range(ModelConstants.IDX_N))

  gated = processor.update(make_inputs(orientation_z=orientation_z))
  held_once = processor.update(make_inputs(previous_desired_curvature=gated.desired_curvature))
  held_twice = processor.update(make_inputs(previous_desired_curvature=held_once.desired_curvature))
  recovered = processor.update(make_inputs(previous_desired_curvature=held_twice.desired_curvature))

  assert gated.gated
  assert gated.reason == "path_disagreement"
  assert gated.hold_frames_remaining == 2
  assert held_once.gated
  assert held_once.reason == "path_disagreement"
  assert held_once.hold_frames_remaining == 1
  assert held_twice.gated
  assert held_twice.reason == "path_disagreement"
  assert held_twice.hold_frames_remaining == 0
  assert not recovered.gated
  assert recovered.reason == "ok"
  assert recovered.hold_frames_remaining == 0


def test_frame_drop_gates_then_recovers_after_clean_hold():
  processor = ModelPathProcessor()

  gated = processor.update(make_inputs(frame_drop_perc=50.0))
  held_once = processor.update(make_inputs(previous_desired_curvature=gated.desired_curvature))
  held_twice = processor.update(make_inputs(previous_desired_curvature=held_once.desired_curvature))
  recovered = processor.update(make_inputs(previous_desired_curvature=held_twice.desired_curvature))

  assert gated.gated
  assert gated.reason == "frame_drop"
  assert gated.hold_frames_remaining == 2
  assert held_once.gated
  assert held_once.reason == "frame_drop"
  assert held_once.hold_frames_remaining == 1
  assert held_twice.gated
  assert held_twice.reason == "frame_drop"
  assert held_twice.hold_frames_remaining == 0
  assert not recovered.gated
  assert recovered.reason == "ok"
  assert recovered.hold_frames_remaining == 0
