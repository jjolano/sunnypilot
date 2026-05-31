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
    self.valid = {'modelDataV2SP': False}

  def all_checks(self, services):
    assert services == ['lateralManeuverPlan']
    return self._checks_ok


def make_controls(*, desired_curvature=0.001, checks_ok=True):
  controls = Controls.__new__(Controls)
  controls.sm = FakeSubMaster(desired_curvature=desired_curvature, checks_ok=checks_ok)
  return controls


class FakeParams:
  def get_bool(self, _key):
    return False


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
  return controls


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

  demand = controls.build_processed_lateral_demand(make_car_control(), CS, make_model_v2(raw_curvature), live_params)
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
  assert demand.lane_change_shaping_active
  assert demand.lane_change_blend == pytest.approx(0.5)
  assert demand.lateral_accel_limit == pytest.approx(MAX_LATERAL_ACCEL_NO_ROLL)
  assert controls.desired_curvature == pytest.approx(expected_curvature)
  assert controls.model_path_raw_desired_curvature == pytest.approx(raw_curvature)
  assert controls.model_path_processor.inputs.desired_curvature == pytest.approx(raw_curvature)
  assert controls.lane_change_path_shaper.inputs.model_curvature == pytest.approx(path_result.desired_curvature)


def test_lateral_maneuver_plan_processed_demand_resets_path_and_lane_shaping():
  maneuver_curvature = 0.008
  raw_curvature = -0.02
  controls = make_demand_controls(desired_curvature=maneuver_curvature, checks_ok=True)
  CS = make_car_state(v_ego=10.0)
  live_params = make_live_params()

  demand = controls.build_processed_lateral_demand(make_car_control(), CS, make_model_v2(raw_curvature), live_params)
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

  demand = controls.build_processed_lateral_demand(CC, CS, make_model_v2(0.02), live_params)
  expected_curvature, expected_limited, _ = clip_curvature(
    CS.vEgo,
    previous_curvature,
    measured_curvature,
    live_params.roll,
    MAX_LATERAL_ACCEL_NO_ROLL,
  )

  assert controls.model_path_processor.inputs.lat_active is False
  assert demand.path_reason == "inactive"
  assert demand.processed_curvature == pytest.approx(expected_curvature)
  assert demand.processed_curvature != pytest.approx(measured_curvature)
  assert demand.curvature_limited == expected_limited
  assert controls.desired_curvature == pytest.approx(expected_curvature)
