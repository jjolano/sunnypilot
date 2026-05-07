from types import SimpleNamespace
import sys
import types

import pytest

visionipc = types.ModuleType("msgq.visionipc")
visionipc.VisionBuf = object
visionipc.VisionIpcClient = object
visionipc.VisionIpcServer = object
visionipc.VisionStreamType = object
visionipc.get_endpoint_name = lambda *args, **kwargs: ""
sys.modules.setdefault("msgq.visionipc", visionipc)

from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.lib.experimental_lateral_path_planner import (
  ExperimentalLateralPathPlannerResult,
  ExperimentalLateralPathPlannerState,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


class FakeParams:
  def __init__(self, enabled: bool):
    self.enabled = enabled

  def get_bool(self, key: str) -> bool:
    assert key == "ExperimentalLateralPathPlanner"
    return self.enabled


class RecordingPlanner:
  def __init__(self, desired_curvature: float):
    self.desired_curvature = desired_curvature
    self.last_inputs = None
    self.reset_count = 0

  def reset(self):
    self.reset_count += 1

  def update(self, inputs):
    self.last_inputs = inputs
    active = inputs.enabled and inputs.lat_active and not inputs.lane_change_active
    desired_curvature = self.desired_curvature if active else inputs.baseline_curvature
    return ExperimentalLateralPathPlannerResult(
      desired_curvature=desired_curvature,
      candidate_curvature=desired_curvature,
      baseline_curvature=inputs.baseline_curvature,
      confidence=1.0 if active else 0.0,
      active=active,
      state=ExperimentalLateralPathPlannerState.active if active else ExperimentalLateralPathPlannerState.baseline,
      reason="ok" if active else "disabled",
    )


def make_model_v2():
  return SimpleNamespace(
    position=SimpleNamespace(
      x=tuple(float(i) for i in range(ModelConstants.IDX_N)),
      y=tuple(0.1 for _ in range(ModelConstants.IDX_N)),
      yStd=tuple(0.2 for _ in range(ModelConstants.IDX_N)),
    ),
    laneLineProbs=(0.0, 0.9, 0.8, 0.0),
    laneLines=(
      SimpleNamespace(y=()),
      SimpleNamespace(y=(-1.7,)),
      SimpleNamespace(y=(1.9,)),
      SimpleNamespace(y=()),
    ),
    roadEdges=(
      SimpleNamespace(y=(-3.2,)),
      SimpleNamespace(y=(3.3,)),
    ),
  )


def make_controls(*, enabled: bool, planner_curvature: float = 0.0025):
  controls = Controls.__new__(Controls)
  controls.params = FakeParams(enabled)
  controls.experimental_lateral_path_planner = RecordingPlanner(planner_curvature)
  controls.curvature = 0.0007
  controls.desired_curvature = 0.001
  return controls


def test_experimental_lateral_path_planner_toggle_off_returns_baseline_and_resets_planner():
  controls = make_controls(enabled=False, planner_curvature=0.0025)

  result = controls.apply_experimental_lateral_path_planner(True, 20.0, make_model_v2(), 0.0012, lane_change_active=False)

  assert result == pytest.approx(0.0012)
  assert controls.experimental_lateral_path_planner.last_inputs.enabled is False
  assert controls.experimental_lateral_path_planner.last_inputs.baseline_curvature == pytest.approx(0.0012)


def test_experimental_lateral_path_planner_toggle_on_passes_model_context_and_uses_result():
  controls = make_controls(enabled=True, planner_curvature=0.0025)

  result = controls.apply_experimental_lateral_path_planner(True, 20.0, make_model_v2(), 0.0012, lane_change_active=False)

  inputs = controls.experimental_lateral_path_planner.last_inputs
  assert result == pytest.approx(0.0025)
  assert inputs.enabled is True
  assert inputs.lat_active is True
  assert inputs.v_ego == pytest.approx(20.0)
  assert inputs.baseline_curvature == pytest.approx(0.0012)
  assert inputs.measured_curvature == pytest.approx(0.0007)
  assert inputs.previous_desired_curvature == pytest.approx(0.001)
  assert inputs.position_x == tuple(float(i) for i in range(ModelConstants.IDX_N))
  assert inputs.position_y == tuple(0.1 for _ in range(ModelConstants.IDX_N))
  assert inputs.position_y_std == tuple(0.2 for _ in range(ModelConstants.IDX_N))
  assert inputs.lane_line_probs == (0.0, 0.9, 0.8, 0.0)
  assert inputs.left_lane_y0 == pytest.approx(-1.7)
  assert inputs.right_lane_y0 == pytest.approx(1.9)
  assert inputs.left_road_edge_y0 == pytest.approx(-3.2)
  assert inputs.right_road_edge_y0 == pytest.approx(3.3)
  assert inputs.lane_change_active is False


def test_experimental_lateral_path_planner_does_not_apply_during_lane_change():
  controls = make_controls(enabled=True, planner_curvature=0.0025)

  result = controls.apply_experimental_lateral_path_planner(True, 20.0, make_model_v2(), 0.0012, lane_change_active=True)

  assert result == pytest.approx(0.0012)
  assert controls.experimental_lateral_path_planner.last_inputs.lane_change_active is True


def test_experimental_lateral_path_planner_allows_missing_road_edges():
  controls = make_controls(enabled=True, planner_curvature=0.0025)
  model_v2 = make_model_v2()
  delattr(model_v2, "roadEdges")

  result = controls.apply_experimental_lateral_path_planner(True, 20.0, model_v2, 0.0012, lane_change_active=False)

  inputs = controls.experimental_lateral_path_planner.last_inputs
  assert result == pytest.approx(0.0025)
  assert inputs.left_road_edge_y0 is None
  assert inputs.right_road_edge_y0 is None
