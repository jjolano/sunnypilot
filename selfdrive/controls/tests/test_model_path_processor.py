import math

import pytest

from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessor, ModelPathProcessorInputs
from openpilot.selfdrive.modeld.constants import ModelConstants


def make_inputs(**overrides) -> ModelPathProcessorInputs:
  data = {
    "lat_active": True,
    "v_ego": 20.0,
    "desired_curvature": 0.002,
    "measured_curvature": 0.0005,
    "previous_desired_curvature": 0.001,
    "position_x": tuple(range(ModelConstants.IDX_N)),
    "position_y": tuple(0.02 * i for i in range(ModelConstants.IDX_N)),
    "position_y_std": tuple(0.1 for _ in range(ModelConstants.IDX_N)),
    "orientation_z": tuple(0.0 for _ in range(ModelConstants.IDX_N)),
    "orientation_rate_z": tuple(0.0 for _ in range(ModelConstants.IDX_N)),
    "lane_line_probs": (0.0, 0.9, 0.9, 0.0),
    "frame_drop_perc": 0.0,
  }
  data.update(overrides)
  return ModelPathProcessorInputs(**data)


def test_good_path_passes_curvature_unchanged():
  result = ModelPathProcessor().update(make_inputs())

  assert not result.gated
  assert result.reason == "ok"
  assert result.quality == pytest.approx(1.0)
  assert result.desired_curvature == pytest.approx(0.002)


def test_inactive_returns_measured_curvature():
  result = ModelPathProcessor().update(make_inputs(lat_active=False))

  assert result.gated
  assert result.reason == "inactive"
  assert result.desired_curvature == pytest.approx(0.0005)


def test_nonfinite_curvature_falls_back_to_previous_desired():
  result = ModelPathProcessor().update(make_inputs(desired_curvature=math.nan))

  assert result.gated
  assert result.reason == "nonfinite_curvature"
  assert result.desired_curvature == pytest.approx(0.001)


def test_missing_path_falls_back_to_previous_desired():
  result = ModelPathProcessor().update(make_inputs(position_x=(0.0, 1.0), position_y=(0.0, 0.1)))

  assert result.gated
  assert result.reason == "invalid_path"
  assert result.desired_curvature == pytest.approx(0.001)


def test_nonfinite_path_sample_falls_back_to_previous_desired():
  position_y = [0.0 for _ in range(ModelConstants.IDX_N)]
  position_y[5] = math.inf

  result = ModelPathProcessor().update(make_inputs(position_y=tuple(position_y)))

  assert result.gated
  assert result.reason == "invalid_path"
  assert result.desired_curvature == pytest.approx(0.001)


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


def test_high_path_std_blends_toward_previous_desired():
  result = ModelPathProcessor().update(make_inputs(position_y_std=tuple(1.4 for _ in range(ModelConstants.IDX_N))))

  assert result.gated
  assert result.reason == "high_path_std"
  assert 0.001 < result.desired_curvature < 0.002


def test_path_curvature_disagreement_blends_toward_previous_desired():
  orientation_z = tuple(0.05 for _ in range(ModelConstants.IDX_N))
  result = ModelPathProcessor().update(make_inputs(orientation_z=orientation_z))

  assert result.gated
  assert result.reason == "path_disagreement"
  assert 0.001 < result.desired_curvature < 0.002
