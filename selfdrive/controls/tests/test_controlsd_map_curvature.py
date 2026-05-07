from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.controlsd import map_curvature_at_model_horizon


def make_live_map_data(**overrides):
  data = {
    "roadCurvatureValid": True,
    "roadCurvatureDistances": [0.0, 10.0],
    "roadCurvatures": [0.0, 0.002],
  }
  data.update(overrides)
  return SimpleNamespace(**data)


def test_map_curvature_at_model_horizon_requires_enabled_toggle():
  curvature = map_curvature_at_model_horizon(make_live_map_data(), v_ego=20.0, enabled=False)

  assert curvature is None


def test_map_curvature_at_model_horizon_interpolates_action_distance():
  curvature = map_curvature_at_model_horizon(make_live_map_data(), v_ego=20.0, enabled=True)

  assert curvature == pytest.approx(0.001)


def test_map_curvature_at_model_horizon_uses_supplied_action_horizon():
  curvature = map_curvature_at_model_horizon(make_live_map_data(), v_ego=20.0, enabled=True, action_t=0.5)

  assert curvature == pytest.approx(0.002)


def test_map_curvature_at_model_horizon_rejects_invalid_map_data():
  curvature = map_curvature_at_model_horizon(
    make_live_map_data(roadCurvatureValid=False),
    v_ego=20.0,
    enabled=True,
  )

  assert curvature is None


@pytest.mark.parametrize("v_ego", [float("nan"), float("inf"), -float("inf")])
def test_map_curvature_at_model_horizon_rejects_non_finite_speed(v_ego):
  curvature = map_curvature_at_model_horizon(
    make_live_map_data(),
    v_ego=v_ego,
    enabled=True,
  )

  assert curvature is None


def test_map_curvature_at_model_horizon_rejects_mismatched_samples():
  curvature = map_curvature_at_model_horizon(
    make_live_map_data(roadCurvatureDistances=[0.0, 10.0], roadCurvatures=[0.0]),
    v_ego=20.0,
    enabled=True,
  )

  assert curvature is None


def test_map_curvature_at_model_horizon_rejects_far_beyond_map_path():
  curvature = map_curvature_at_model_horizon(
    make_live_map_data(roadCurvatureDistances=[0.0], roadCurvatures=[0.001]),
    v_ego=40.0,
    enabled=True,
  )

  assert curvature is None
