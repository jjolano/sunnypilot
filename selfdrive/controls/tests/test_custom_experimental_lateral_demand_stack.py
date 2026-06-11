import types
import pytest

from openpilot.selfdrive.controls.lib.lateral_demand import (
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)
from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfile
from openpilot.selfdrive.controls.lib.lateral_demand_stacks import (
  LateralDemandStackInputs,
  LateralDemandStackOutput,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_experimental import CustomExperimentalLateralDemandStack
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessorResult


def _model_v2():
  zeros = (0.0,) * 33
  return types.SimpleNamespace(
    position=types.SimpleNamespace(x=zeros, y=zeros, xStd=zeros, yStd=zeros),
    orientation=types.SimpleNamespace(z=zeros),
    orientationRate=types.SimpleNamespace(z=zeros),
    laneLineProbs=(0.0, 0.9, 0.9, 0.0),
    laneLineStds=(0.0, 0.0, 0.0, 0.0),
    laneLines=[types.SimpleNamespace(y=[]), types.SimpleNamespace(y=[-1.8]),
               types.SimpleNamespace(y=[1.8]), types.SimpleNamespace(y=[])],
    frameDropPerc=0.0,
  )


def _live_params():
  return types.SimpleNamespace(roll=0.0)


def _build_inputs(v_ego=20.0, lat_active=True):
  return LateralDemandStackInputs(
    lat_active=lat_active,
    v_ego=v_ego,
    desired_curvature=0.002,
    measured_curvature=0.0015,
    model_v2=_model_v2(),
    live_params=_live_params(),
    curvature_limited=False,
    accurate_lateral_accel=False,
    manual_gas_lateral_accel_override=False,
    lateral_maneuver_curvature=None,
    roll=0.0,
    lateral_accel_limit_no_roll=2.5,
    default_lateral_accel_limited=False,
    lane_change_state=0,
    lane_change_direction=0,
    turn_direction=0,
    model_data_v2_sp_valid=True,
    lane_centering_assist_enabled=False,
    gas_pressed=False,
    brake_pressed=False,
    steering_pressed=False,
    left_blinker=False,
    right_blinker=False,
    left_lane_y0=-1.8,
    right_lane_y0=1.8,
    frame_drop_perc=0.0,
    smoothed_model_path_curvature=False,
  )


def _make_v2(path_result):
  stack = CustomV2LateralDemandStack(dt=0.05)
  stack._model_path_result = path_result
  return stack


def test_custom_experimental_matches_v2_legacy_curvature():
  v2_path_result = ModelPathProcessorResult(0.0015, 0.9, False, "ok")
  v2_stack = _make_v2(v2_path_result)
  exp_stack = CustomExperimentalLateralDemandStack(dt=0.05)
  inputs = _build_inputs(v_ego=20.0, lat_active=True)
  v2_output = v2_stack.update(inputs)
  exp_output = exp_stack.update(inputs)
  assert isinstance(v2_output.legacy, ProcessedLateralDemand)
  assert isinstance(exp_output.legacy, ProcessedLateralDemand)
  assert exp_output.legacy.processed_curvature == pytest.approx(v2_output.legacy.processed_curvature)
  assert exp_output.legacy.demand_source == DEMAND_SOURCE_MODEL_PATH
  assert exp_output.legacy.path_reason == v2_output.legacy.path_reason


def test_custom_experimental_inherits_v2_path_processor_state():
  v2_path_result = ModelPathProcessorResult(0.002, 0.7, False, "ok")
  v2_stack = _make_v2(v2_path_result)
  exp_stack = CustomExperimentalLateralDemandStack(dt=0.05)
  inputs = _build_inputs(v_ego=20.0, lat_active=True)
  v2_stack.update(inputs)
  exp_stack.update(inputs)
  assert exp_stack.model_path_result.reason == v2_stack.model_path_result.reason
  assert exp_stack.model_path_result.quality == pytest.approx(v2_stack.model_path_result.quality)


def test_custom_experimental_output_identifies_itself():
  exp_stack = CustomExperimentalLateralDemandStack(dt=0.05)
  inputs = _build_inputs(v_ego=20.0, lat_active=True)
  output = exp_stack.update(inputs)
  assert isinstance(output, LateralDemandStackOutput)
  assert output.requested_stack == "custom-experimental"
  assert output.resolved_stack == "custom-experimental"
  assert output.version == "experimental"
  assert output.fallback_reason == ""


def test_custom_experimental_reports_stage_v2_baseline():
  exp_stack = CustomExperimentalLateralDemandStack(dt=0.05)
  inputs = _build_inputs(v_ego=20.0, lat_active=True)
  output = exp_stack.update(inputs)
  assert output.debug["experimental_stage"] == "v2_baseline"
  assert exp_stack.stage == "v2_baseline"


def test_custom_experimental_profile_matches_v2_classification():
  v2_path_result = ModelPathProcessorResult(0.001, 1.0, False, "ok")
  v2_stack = _make_v2(v2_path_result)
  exp_stack = CustomExperimentalLateralDemandStack(dt=0.05)
  inputs = _build_inputs(v_ego=20.0, lat_active=True)
  v2_output = v2_stack.update(inputs)
  exp_output = exp_stack.update(inputs)
  assert isinstance(v2_output.profile, LateralDemandProfile)
  assert isinstance(exp_output.profile, LateralDemandProfile)
  assert exp_output.profile.mode == v2_output.profile.mode


def test_custom_experimental_reset_clears_state():
  exp_stack = CustomExperimentalLateralDemandStack(dt=0.05)
  inputs = _build_inputs(v_ego=20.0, lat_active=True)
  exp_stack.update(inputs)
  exp_stack.reset()
  assert exp_stack.model_path_result.reason == "inactive"
  assert exp_stack.last_legacy_demand is None
  assert exp_stack.last_profile is None
  assert exp_stack.stage == "v2_baseline"
