import math
import sys
import types

import pytest

params_pyx = types.ModuleType("openpilot.common.params_pyx")
params_pyx.Params = object
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

msgq = types.ModuleType("msgq")
msgq.fake_event_handle = object()
msgq.drain_sock_raw = lambda *args, **kwargs: []
msgq.MultiplePublishersError = RuntimeError
msgq.IpcError = RuntimeError
msgq.Context = object
msgq.Poller = object
msgq.SubSocket = object
msgq.PubSocket = object
msgq.SocketEventHandle = object
msgq.toggle_fake_events = lambda *args, **kwargs: None
msgq.set_fake_prefix = lambda *args, **kwargs: None
msgq.get_fake_prefix = lambda *args, **kwargs: ""
msgq.delete_fake_prefix = lambda *args, **kwargs: None
msgq.wait_for_one_event = lambda *args, **kwargs: None
msgq.pub_sock = lambda *args, **kwargs: None
msgq.sub_sock = lambda *args, **kwargs: None
msgq.context = None
sys.modules.setdefault("msgq", msgq)

visionipc = types.ModuleType("msgq.visionipc")
visionipc.VisionBuf = object
visionipc.VisionIpcClient = object
visionipc.VisionIpcServer = object
visionipc.VisionStreamType = object
visionipc.get_endpoint_name = lambda *args, **kwargs: ""
sys.modules.setdefault("msgq.visionipc", visionipc)

from cereal import car, log
import cereal.messaging as messaging
from openpilot.selfdrive.controls.controlsd import Controls, fill_model_path_state, model_path_reason_to_capnp
from openpilot.selfdrive.controls.lib.drive_helpers import MAX_LATERAL_ACCEL_NO_ROLL, clip_curvature
from openpilot.selfdrive.controls.lib.lane_change_path_shaper import LaneChangePathShaperResult
from openpilot.selfdrive.controls.lib.lane_centering_assist import LaneCenteringAssistResult, inactive_lane_centering_assist_result
from openpilot.selfdrive.controls.lib.lateral_demand import (
  DEMAND_SOURCE_FALLBACK_MEASURED,
  DEMAND_SOURCE_LATERAL_MANEUVER,
  DEMAND_SOURCE_MODEL_PATH,
  ProcessedLateralDemand,
)
from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfile, LateralDemandProfileBuilder
from openpilot.selfdrive.controls.lib.model_path_processor import ModelPathProcessorResult


class FakeLateralManeuverPlan:
  def __init__(self, desired_curvature):
    self.desiredCurvature = desired_curvature


class FakeSubMaster(dict):
  def __init__(self, *, desired_curvature, checks_ok):
    super().__init__(
      lateralManeuverPlan=FakeLateralManeuverPlan(desired_curvature),
      modelDataV2SP=types.SimpleNamespace(laneTurnDirection=0),
    )
    self._checks_ok = checks_ok
    self.valid = {'modelDataV2SP': False, 'modelV2': True, 'liveParameters': True, 'lateralManeuverPlan': checks_ok}

  def all_checks(self, services):
    assert services == ['lateralManeuverPlan']
    return self._checks_ok


def make_controls(*, desired_curvature=0.001, checks_ok=True):
  controls = Controls.__new__(Controls)
  controls.sm = FakeSubMaster(desired_curvature=desired_curvature, checks_ok=checks_ok)
  return controls


class FakeParams:
  def __init__(self):
    self._store = {}

  def get_bool(self, key):
    return self._store.get(key, False)

  def put_bool(self, key, value):
    self._store[key] = bool(value)


class FakeModelPathProcessor:
  def __init__(self, result):
    self.result = result
    self.inputs = None
    self.reset_count = 0

  def update(self, inputs):
    self.inputs = inputs
    return self.result

  def reset(self):
    self.reset_count += 1


class FakeLaneChangePathShaper:
  def __init__(self, result):
    self.result = result
    self.inputs = None
    self.reset_count = 0

  def update(self, inputs):
    self.inputs = inputs
    return self.result

  def reset(self):
    self.reset_count += 1


class FakeLaneCenteringAssistTracker:
  def __init__(self, result=None):
    self.result = result or inactive_lane_centering_assist_result()
    self.inputs = None
    self.dt = None
    self.reset_count = 0

  def update(self, inputs, dt):
    self.inputs = inputs
    self.dt = dt
    return self.result

  def reset(self):
    self.reset_count += 1


def make_model_v2(raw_curvature=0.002):
  samples = [float(i) for i in range(33)]
  zeros = [0.0 for _ in samples]
  return types.SimpleNamespace(
    action=types.SimpleNamespace(desiredCurvature=raw_curvature),
    meta=types.SimpleNamespace(laneChangeState=log.LaneChangeState.off, laneChangeDirection=log.LaneChangeDirection.none),
    position=types.SimpleNamespace(x=samples, y=zeros, yStd=zeros),
    orientation=types.SimpleNamespace(z=zeros),
    orientationRate=types.SimpleNamespace(z=zeros),
    laneLineProbs=(0.0, 0.9, 0.9, 0.0),
    laneLineStds=(0.0, 0.0, 0.0, 0.0),
    laneLines=[types.SimpleNamespace(y=[]), types.SimpleNamespace(y=[-1.8]), types.SimpleNamespace(y=[1.8]), types.SimpleNamespace(y=[])],
    frameDropPerc=0.0,
  )


def make_car_control(lat_active=True, long_active=True):
  CC = car.CarControl.new_message()
  CC.latActive = lat_active
  CC.longActive = long_active
  return CC


def make_car_state(v_ego=10.0):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.leftBlinker = False
  CS.rightBlinker = False
  CS.steeringPressed = False
  CS.gasPressed = False
  CS.brakePressed = False
  return CS


def make_live_params(roll=0.0):
  live_params = log.LiveParametersData.new_message()
  live_params.roll = roll
  return live_params


def make_demand_controls(*, previous_curvature=0.0, measured_curvature=0.001, desired_curvature=0.001,
                         checks_ok=False, path_result=None, lane_result=None):
  controls = make_controls(desired_curvature=desired_curvature, checks_ok=checks_ok)
  controls.params = FakeParams()
  controls.curvature = measured_curvature
  controls.desired_curvature = previous_curvature
  controls.lateral_accel_limit_no_roll = MAX_LATERAL_ACCEL_NO_ROLL
  controls.default_lateral_accel_limited = False
  controls.smoothed_model_path_curvature = False
  controls.model_path_result = ModelPathProcessorResult(0.0, 0.0, True, "inactive")
  controls.model_path_raw_desired_curvature = 0.0
  controls.model_path_processor = FakeModelPathProcessor(path_result or ModelPathProcessorResult(measured_curvature, 0.0, True, "inactive"))
  controls.lane_change_path_shaper = FakeLaneChangePathShaper(lane_result or LaneChangePathShaperResult(measured_curvature, 0.0, False, False))
  controls.params.put_bool("LaneCenteringAssistEnabled", False)
  controls.lane_centering_assist_tracker = FakeLaneCenteringAssistTracker()

  from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
  stack = CustomV2LateralDemandStack(dt=0.05)
  stack._model_path_processor = controls.model_path_processor
  stack._lane_change_path_shaper = controls.lane_change_path_shaper
  stack._lane_centering_assist_tracker = controls.lane_centering_assist_tracker
  stack._model_path_result = controls.model_path_result
  stack._smoothed_model_path_curvature = controls.smoothed_model_path_curvature
  stack._previous_desired_curvature = previous_curvature
  stack._lateral_accel_limit_no_roll = controls.lateral_accel_limit_no_roll
  stack._default_lateral_accel_limited = controls.default_lateral_accel_limited
  controls.lateral_demand_stack = stack
  return controls


def run_lateral_demand(controls, CC, CS, model_v2, live_params):
  from openpilot.selfdrive.controls.lib.lateral_demand_stacks import LateralDemandStackInputs
  if hasattr(controls, 'lane_centering_assist_tracker') and controls.lane_centering_assist_tracker is not controls.lateral_demand_stack._lane_centering_assist_tracker:
    controls.lateral_demand_stack._lane_centering_assist_tracker = controls.lane_centering_assist_tracker
  if hasattr(controls, 'model_path_processor') and controls.model_path_processor is not controls.lateral_demand_stack._model_path_processor:
    controls.lateral_demand_stack._model_path_processor = controls.model_path_processor
  if hasattr(controls, 'lane_change_path_shaper') and controls.lane_change_path_shaper is not controls.lateral_demand_stack._lane_change_path_shaper:
    controls.lateral_demand_stack._lane_change_path_shaper = controls.lane_change_path_shaper
  stack_inputs = controls.build_lateral_demand_stack_inputs(CC, CS, model_v2, live_params)
  output = controls.lateral_demand_stack.update(stack_inputs)
  controls.desired_curvature = output.legacy.processed_curvature
  controls.processed_lateral_demand = output.legacy
  controls.model_path_result = controls.lateral_demand_stack.model_path_result
  controls.model_path_raw_desired_curvature = controls.lateral_demand_stack.model_path_raw_desired_curvature
  controls.smoothed_model_path_curvature = controls.lateral_demand_stack.smoothed_model_path_curvature
  controls.lateral_accel_limit_no_roll = controls.lateral_demand_stack.lateral_accel_limit_no_roll
  controls.default_lateral_accel_limited = controls.lateral_demand_stack.default_lateral_accel_limited
  return output


def test_lateral_maneuver_curvature_uses_fresh_finite_plan():
  controls = make_controls(desired_curvature=0.0015, checks_ok=True)

  assert controls.get_lateral_maneuver_curvature(True) == pytest.approx(0.0015)


def test_lateral_maneuver_curvature_ignores_stale_plan():
  controls = make_controls(desired_curvature=0.0015, checks_ok=False)

  assert controls.get_lateral_maneuver_curvature(True) is None


@pytest.mark.parametrize("desired_curvature", [math.nan, math.inf, -math.inf])
def test_lateral_maneuver_curvature_ignores_nonfinite_plan(desired_curvature):
  controls = make_controls(desired_curvature=desired_curvature, checks_ok=True)

  assert controls.get_lateral_maneuver_curvature(True) is None


def test_lateral_maneuver_curvature_ignores_plan_when_lateral_inactive():
  controls = make_controls(desired_curvature=0.0015, checks_ok=True)

  assert controls.get_lateral_maneuver_curvature(False) is None


def test_model_path_reason_mapping_uses_controls_state_schema_enum():
  assert model_path_reason_to_capnp("path_disagreement") == log.ControlsState.ModelPathState.Reason.pathDisagreement
  assert model_path_reason_to_capnp("unexpected") == log.ControlsState.ModelPathState.Reason.unknown


def test_fill_model_path_state_publishes_processed_path_debug_values():
  msg = messaging.new_message('controlsState')
  result = ModelPathProcessorResult(
    0.0015,
    0.65,
    True,
    "path_disagreement",
    1,
    smoothing_tau_s=0.11,
    damping_alpha=0.18,
    trust_penalty=0.4,
    spatial_smoothed_curvature=0.00155,
    lane_change_fade=0.75,
  )

  fill_model_path_state(msg.controlsState.modelPathState, result, 0.002)

  state = msg.controlsState.modelPathState
  assert state.active
  assert state.gated
  assert state.quality == pytest.approx(0.65)
  assert state.reason == log.ControlsState.ModelPathState.Reason.pathDisagreement
  assert state.rawDesiredCurvature == pytest.approx(0.002)
  assert state.processedDesiredCurvature == pytest.approx(0.0015)
  assert state.holdFramesRemaining == 1
  assert state.smoothingTauS == pytest.approx(0.11)
  assert state.dampingAlpha == pytest.approx(0.18)
  assert state.trustPenalty == pytest.approx(0.4)
  assert state.spatialSmoothedCurvature == pytest.approx(0.00155)
  assert state.laneChangeFade == pytest.approx(0.75)


def test_processed_lateral_demand_tracks_raw_path_processed_and_clipped_curvature():
  path_result = ModelPathProcessorResult(0.006, 0.8, False, "ok")
  lane_result = LaneChangePathShaperResult(0.008, 0.5, True, False)
  controls = make_demand_controls(path_result=path_result, lane_result=lane_result)
  CS = make_car_state(v_ego=10.0)
  live_params = make_live_params()
  raw_curvature = 0.02

  output = run_lateral_demand(controls, make_car_control(), CS, make_model_v2(raw_curvature), live_params)
  demand = output.legacy
  expected_curvature, expected_limited, _ = clip_curvature(
    CS.vEgo,
    0.0,
    lane_result.desired_curvature,
    live_params.roll,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert demand.raw_curvature == pytest.approx(raw_curvature)
  assert demand.processed_curvature == pytest.approx(expected_curvature)
  assert demand.curvature_limited == expected_limited
  assert demand.measured_curvature == pytest.approx(controls.curvature)
  assert demand.path_quality == pytest.approx(path_result.quality)
  assert demand.path_reason == path_result.reason
  assert demand.demand_source == DEMAND_SOURCE_MODEL_PATH
  assert demand.lane_change_shaping_active
  assert demand.lane_change_blend == pytest.approx(0.5)
  assert demand.lateral_accel_limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)
  assert controls.desired_curvature == pytest.approx(expected_curvature)
  assert controls.model_path_raw_desired_curvature == pytest.approx(raw_curvature)
  assert controls.model_path_processor.inputs.desired_curvature == pytest.approx(raw_curvature)
  assert controls.lane_change_path_shaper.inputs.model_curvature == pytest.approx(path_result.desired_curvature)
  assert not demand.lane_centering_assist_active
  assert demand.lane_centering_curvature_nudge == pytest.approx(0.0)


def test_lane_centering_assist_is_default_off_for_processed_demand():
  path_result = ModelPathProcessorResult(0.001, 1.0, False, "ok")
  lane_result = LaneChangePathShaperResult(0.001, 0.0, False, False)
  controls = make_demand_controls(path_result=path_result, lane_result=lane_result)
  controls.lane_centering_assist_tracker = FakeLaneCenteringAssistTracker(
    LaneCenteringAssistResult(True, 0.0005, 0.1, 0.0, 0.2, 1.0, "growing_lateral_error")
  )
  CS = make_car_state(v_ego=20.0)
  live_params = make_live_params()

  output = run_lateral_demand(controls, make_car_control(), CS, make_model_v2(), live_params)
  demand = output.legacy
  expected_curvature, _expected_limited, _ = clip_curvature(
    CS.vEgo, 0.0, lane_result.desired_curvature, live_params.roll, MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert controls.lane_centering_assist_tracker.inputs is None
  assert demand.processed_curvature == pytest.approx(expected_curvature)
  assert not demand.lane_centering_assist_active


def test_lane_centering_assist_nudges_before_final_clipping_when_enabled():
  path_result = ModelPathProcessorResult(0.001, 1.0, False, "ok")
  lane_result = LaneChangePathShaperResult(0.001, 0.0, False, False)
  controls = make_demand_controls(path_result=path_result, lane_result=lane_result)
  controls.params.put_bool("LaneCenteringAssistEnabled", True)
  controls.lane_centering_assist_tracker = FakeLaneCenteringAssistTracker(
    LaneCenteringAssistResult(True, 0.0002, 0.1, 0.0, 0.2, 1.0, "growing_lateral_error")
  )
  CS = make_car_state(v_ego=20.0)
  live_params = make_live_params()

  output = run_lateral_demand(controls, make_car_control(), CS, make_model_v2(), live_params)
  demand = output.legacy
  expected_curvature, _expected_limited, _ = clip_curvature(
    CS.vEgo, 0.0, lane_result.desired_curvature + 0.0002, live_params.roll, MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert controls.lane_centering_assist_tracker.inputs is not None
  assert controls.lane_centering_assist_tracker.inputs.model_curvature == pytest.approx(lane_result.desired_curvature)
  assert demand.processed_curvature == pytest.approx(expected_curvature)
  assert demand.lane_centering_assist_active
  assert demand.lane_centering_curvature_nudge == pytest.approx(0.0002)


def test_lane_centering_assist_does_not_bypass_final_clipping():
  path_result = ModelPathProcessorResult(0.0, 1.0, False, "ok")
  lane_result = LaneChangePathShaperResult(0.0, 0.0, False, False)
  controls = make_demand_controls(path_result=path_result, lane_result=lane_result)
  controls.params.put_bool("LaneCenteringAssistEnabled", True)
  controls.lane_centering_assist_tracker = FakeLaneCenteringAssistTracker(
    LaneCenteringAssistResult(True, 1.0, 0.1, 0.0, 0.2, 1.0, "growing_lateral_error")
  )
  CS = make_car_state(v_ego=30.0)
  live_params = make_live_params()

  output = run_lateral_demand(controls, make_car_control(), CS, make_model_v2(), live_params)
  demand = output.legacy
  expected_curvature, expected_limited, _ = clip_curvature(
    CS.vEgo, 0.0, 1.0, live_params.roll, MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert demand.processed_curvature == pytest.approx(expected_curvature)
  assert demand.curvature_limited == expected_limited
  assert demand.processed_curvature != pytest.approx(1.0)


def test_lateral_maneuver_source_does_not_call_lane_centering_assist():
  controls = make_demand_controls(desired_curvature=0.002, checks_ok=True)
  controls.params.put_bool("LaneCenteringAssistEnabled", True)
  tracker = FakeLaneCenteringAssistTracker(LaneCenteringAssistResult(True, 0.0002, 0.1, 0.0, 0.2, 1.0, "growing_lateral_error"))
  controls.lane_centering_assist_tracker = tracker

  output = run_lateral_demand(controls, make_car_control(), make_car_state(), make_model_v2(), make_live_params())
  demand = output.legacy
  assert tracker.inputs is None
  assert demand.demand_source == DEMAND_SOURCE_LATERAL_MANEUVER
  assert not demand.lane_centering_assist_active


def test_lateral_maneuver_plan_processed_demand_resets_path_and_lane_shaping():
  maneuver_curvature = 0.008
  raw_curvature = -0.02
  controls = make_demand_controls(desired_curvature=maneuver_curvature, checks_ok=True)
  CS = make_car_state(v_ego=10.0)
  live_params = make_live_params()

  output = run_lateral_demand(controls, make_car_control(), CS, make_model_v2(raw_curvature), live_params)
  demand = output.legacy
  expected_curvature, expected_limited, _ = clip_curvature(
    CS.vEgo,
    0.0,
    maneuver_curvature,
    live_params.roll,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert demand.raw_curvature == pytest.approx(raw_curvature)
  assert demand.processed_curvature == pytest.approx(expected_curvature)
  assert demand.curvature_limited == expected_limited
  assert demand.path_reason == "lateral_maneuver"
  assert demand.demand_source == DEMAND_SOURCE_LATERAL_MANEUVER
  assert demand.path_quality == pytest.approx(0.0)
  assert not demand.lane_change_shaping_active
  assert demand.lane_change_blend == pytest.approx(0.0)
  assert controls.model_path_processor.reset_count == 1
  assert controls.lane_change_path_shaper.reset_count == 1
  assert controls.model_path_processor.inputs is None
  assert controls.lane_change_path_shaper.inputs is None
  assert controls.model_path_raw_desired_curvature == pytest.approx(raw_curvature)
  assert controls.desired_curvature == pytest.approx(expected_curvature)


def test_inactive_processed_lateral_demand_preserves_existing_clipping_path():
  measured_curvature = 0.002
  previous_curvature = 0.010
  controls = make_demand_controls(previous_curvature=previous_curvature, measured_curvature=measured_curvature)
  CC = make_car_control(lat_active=False)
  CS = make_car_state(v_ego=10.0)
  live_params = make_live_params()

  output = run_lateral_demand(controls, CC, CS, make_model_v2(0.02), live_params)
  demand = output.legacy
  expected_curvature, expected_limited, _ = clip_curvature(
    CS.vEgo,
    previous_curvature,
    measured_curvature,
    live_params.roll,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert controls.model_path_processor.inputs.lat_active is False
  assert demand.path_reason == "inactive"
  assert demand.demand_source == DEMAND_SOURCE_FALLBACK_MEASURED
  assert demand.processed_curvature == pytest.approx(expected_curvature)
  assert demand.processed_curvature != pytest.approx(measured_curvature)
  assert demand.curvature_limited == expected_limited
  assert controls.desired_curvature == pytest.approx(expected_curvature)


def test_update_lateral_controller_demand_calls_direct_hook():
  class FakeController:
    def __init__(self):
      self.demand = None

    def set_processed_lateral_demand(self, demand):
      self.demand = demand

  demand = ProcessedLateralDemand(0.1, 0.2, 0.3, False, 1.0, "ok", False, 0.0, MAX_LATERAL_ACCEL_NO_ROLL)
  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()

  controls.update_lateral_controller_demand(demand)

  assert controls.LaC.demand is demand


def test_update_lateral_controller_demand_uses_extension_hook():
  class FakeExtension:
    def __init__(self):
      self.demand = None

    def set_processed_lateral_demand(self, demand):
      self.demand = demand

  class FakeController:
    def __init__(self):
      self.extension = FakeExtension()

  demand = ProcessedLateralDemand(0.1, 0.2, 0.3, False, 1.0, "ok", False, 0.0, MAX_LATERAL_ACCEL_NO_ROLL)
  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()

  controls.update_lateral_controller_demand(demand)

  assert controls.LaC.extension.demand is demand


def test_update_lateral_controller_demand_ignores_missing_hook():
  controls = Controls.__new__(Controls)
  controls.LaC = object()
  demand = ProcessedLateralDemand(0.1, 0.2, 0.3, False, 1.0, "ok", False, 0.0, MAX_LATERAL_ACCEL_NO_ROLL)

  controls.update_lateral_controller_demand(demand)


def test_update_lateral_demand_profile_calls_direct_hook():

  class FakeController:
    def __init__(self):
      self.profile = None

    def set_lateral_demand_profile(self, profile):
      self.profile = profile

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
  controls.lateral_demand_stack = CustomV2LateralDemandStack(dt=0.05)
  controls.lateral_demand_stack._lateral_demand_profile_builder = LateralDemandProfileBuilder(dt=0.05)
  demand = ProcessedLateralDemand(0.001, 0.001, 0.0, False, 1.0, "ok", False, 0.0, MAX_LATERAL_ACCEL_NO_ROLL)

  controls.update_lateral_demand_profile(demand, v_ego=20.0)

  assert isinstance(controls.LaC.profile, LateralDemandProfile)
  assert controls.LaC.profile.mode in {
    "straight_stable", "steady_curve", "turn_in", "turn_exit_recenter",
    "lane_change", "low_quality_path", "safety_limited", "driver_override",
  }


def test_update_lateral_demand_profile_uses_extension_hook():
  class FakeExtension:
    def __init__(self):
      self.profile = None

    def set_lateral_demand_profile(self, profile):
      self.profile = profile

  class FakeController:
    def __init__(self):
      self.extension = FakeExtension()

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
  controls.lateral_demand_stack = CustomV2LateralDemandStack(dt=0.05)
  controls.lateral_demand_stack._lateral_demand_profile_builder = LateralDemandProfileBuilder(dt=0.05)
  demand = ProcessedLateralDemand(0.001, 0.001, 0.0, False, 1.0, "ok", False, 0.0, MAX_LATERAL_ACCEL_NO_ROLL)

  controls.update_lateral_demand_profile(demand, v_ego=20.0)

  assert controls.LaC.extension.profile is not None


def test_update_lateral_demand_profile_ignores_missing_hook():
  controls = Controls.__new__(Controls)
  controls.LaC = object()
  from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
  controls.lateral_demand_stack = CustomV2LateralDemandStack(dt=0.05)
  controls.lateral_demand_stack._lateral_demand_profile_builder = LateralDemandProfileBuilder(dt=0.05)
  demand = ProcessedLateralDemand(0.001, 0.001, 0.0, False, 1.0, "ok", False, 0.0, MAX_LATERAL_ACCEL_NO_ROLL)

  controls.update_lateral_demand_profile(demand, v_ego=20.0)


def test_update_lateral_demand_profile_classifies_steering_pressed_as_driver_override():
  from openpilot.selfdrive.controls.lib.lateral_demand_profile import LateralDemandProfileBuilder, LateralMode

  class FakeController:
    def __init__(self):
      self.profile = None

    def set_lateral_demand_profile(self, profile):
      self.profile = profile

  controls = Controls.__new__(Controls)
  controls.LaC = FakeController()
  from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack
  controls.lateral_demand_stack = CustomV2LateralDemandStack(dt=0.05)
  controls.lateral_demand_stack._lateral_demand_profile_builder = LateralDemandProfileBuilder(dt=0.05)
  demand = ProcessedLateralDemand(0.001, 0.001, 0.0, False, 1.0, "ok", False, 0.0, MAX_LATERAL_ACCEL_NO_ROLL)

  controls.update_lateral_demand_profile(demand, v_ego=20.0, steering_pressed=True)

  assert controls.LaC.profile.mode == LateralMode.DRIVER_OVERRIDE.value
