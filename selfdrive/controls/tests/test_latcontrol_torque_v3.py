import math
import pytest

from cereal import log


def test_v3_adaptive_torque_schema_fields_exist():
  torque_log = log.ControlsState.LateralTorqueState.new_message()
  adaptive_log = torque_log.init('adaptiveTorqueState')

  adaptive_log.modelMode = 2
  adaptive_log.modelConfidence = 0.5
  adaptive_log.authorityBand = 1
  adaptive_log.authorityScale = 0.65
  adaptive_log.fallbackActive = False
  adaptive_log.learnedLatAccelFactor = 2.5
  adaptive_log.learnedFriction = 0.1
  adaptive_log.learnedLatAccelOffset = 0.0
  adaptive_log.learnedResponseDelay = 0.2
  adaptive_log.residualError = 0.05
  adaptive_log.sampleAccepted = True
  adaptive_log.sampleRejectReason = 0

  assert adaptive_log.modelMode == 2
  assert adaptive_log.authorityBand == 1
  assert adaptive_log.sampleAccepted


import numpy as np
import sys
import types

from cereal import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.gm.values import CAR as GM
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.locationd.helpers import Measurement, Pose
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_estimator import EstimatorRejectReason, EstimatorResult
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_authority import AuthorityBand
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_model import TorqueModelMode, TorqueModelParams
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_safety import TorqueV3SafetyResult
from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import ConservativeOutputShaperResult, ConservativeOutputShapingReason
from openpilot.sunnypilot.selfdrive.controls.lib.torque_guarded_response_assist import GuardedResponseReason
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import SteeringActuatorFeedback, SteeringLimitReason

params_pyx = types.ModuleType("openpilot.common.params_pyx")


class FakeParams:
  def get_bool(self, _key: str) -> bool:
    return False

  def remove(self, _key: str) -> None:
    pass

  def get(self, _key: str, *_args, **_kwargs):
    return None


params_pyx.Params = FakeParams
params_pyx.ParamKeyFlag = object
params_pyx.ParamKeyType = object
params_pyx.UnknownKeyName = RuntimeError
sys.modules.setdefault("openpilot.common.params_pyx", params_pyx)

from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_v3

LatControlTorque = latcontrol_torque_v3.LatControlTorque
LatControlTorqueV3 = latcontrol_torque_v3.LatControlTorqueV3


def get_controller(car_name, force_pid=False):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CP_SP.neuralNetworkLateralControl.model.path = MOCK_MODEL_PATH
  CI = CarInterface(CP, CP_SP)
  if force_pid:
    CP.lateralTuning.init('pid')
    CP.lateralTuning.pid.kpBP = [0.0]
    CP.lateralTuning.pid.kpV = [0.1]
    CP.lateralTuning.pid.kiBP = [0.0]
    CP.lateralTuning.pid.kiV = [0.01]
  CP_SP = convert_to_capnp(CP_SP)
  VM = VehicleModel(CP)
  controller = LatControlTorque(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)
  return controller, VM


def make_pose():
  zeros = np.zeros(3)
  return Pose(Measurement(zeros, zeros), Measurement(zeros, zeros), Measurement(zeros, zeros), Measurement(zeros, zeros))


def make_flat_model_v2():
  model_v2 = log.ModelDataV2.new_message()
  zeros = [0.0 for _ in ModelConstants.T_IDXS]

  orientation = log.XYZTData.new_message()
  orientation.x = zeros
  orientation.y = zeros
  model_v2.orientation = orientation

  acceleration = log.XYZTData.new_message()
  acceleration.y = zeros
  model_v2.acceleration = acceleration

  return model_v2


class FlatNNTorqueModel:
  friction_override = False

  def evaluate(self, _input_array):
    return 0.0


def enable_flat_nnlc(controller):
  controller.extension.enabled = True
  controller.extension.has_nn_model = True
  controller.extension.model = FlatNNTorqueModel()
  controller.extension.update_model_v2(make_flat_model_v2())


def test_v3_uses_crawl_speed_for_low_speed_pid_gain():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 1.0
  params = log.LiveParametersData.new_message()

  controller.update(True, CS, VM, params, False, 0.0, make_pose(), False, 0.2)

  assert controller.pid.speed == pytest.approx(3.0)


def test_v3_nnlc_uses_crawl_speed_for_low_speed_pid_gain():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  enable_flat_nnlc(controller)
  CS = car.CarState.new_message()
  CS.vEgo = 1.0
  params = log.LiveParametersData.new_message()

  controller.update(True, CS, VM, params, False, 0.0, make_pose(), False, 0.2)

  assert controller.extension._pid.speed == pytest.approx(3.0)


def test_v3_controller_alias_matches_controller_symbol():
  assert LatControlTorqueV3 is LatControlTorque


def test_v3_signed_steer_limit_reaches_safety_shaper():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 5.0
  CS.steeringPressed = False
  CS.steeringRateDeg = 40.0
  params = log.LiveParametersData.new_message()
  pose = make_pose()

  class SpyShaper:
    def __init__(self):
      self.inputs = None

    def update(self, inputs):
      self.inputs = inputs
      return ConservativeOutputShaperResult(inputs.unshaped_output, False, 0, 0.0, inputs.unshaped_output, 1.0)

  spy = SpyShaper()
  controller.safety_envelope.output_shaper = spy
  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, 0.7, 0.45, 0.25, False, True)
  )

  controller.update(True, CS, VM, params, True, 0.005, pose, False, 0.2)

  assert spy.inputs is not None
  assert not spy.inputs.steer_limit_same_direction
  assert spy.inputs.steer_limit_unwind


def test_v3_logs_signed_steer_limit_feedback():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 5.0
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()
  pose = make_pose()

  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, 0.7, 0.45, 0.25, False, True)
  )
  _, _, lac_log = controller.update(True, CS, VM, params, True, 0.005, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.steerLimitValid
  assert adaptive_log.steerLimitLimited
  assert adaptive_log.steerLimitReason == SteeringLimitReason.ACTUATOR_MISMATCH
  assert adaptive_log.steerLimitRequested == pytest.approx(0.7)
  assert adaptive_log.steerLimitApplied == pytest.approx(0.45)
  assert adaptive_log.steerLimitError == pytest.approx(0.25)
  assert not adaptive_log.steerLimitSameDirection
  assert adaptive_log.steerLimitUnwind


def test_v3_signed_unwind_steer_limit_does_not_freeze_response_assist_but_rejects_estimator_sample():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 5.0
  CS.steeringPressed = False
  CS.steeringRateDeg = 40.0
  params = log.LiveParametersData.new_message()
  pose = make_pose()

  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, 0.7, 0.45, 0.25, False, True)
  )
  _, _, lac_log = controller.update(True, CS, VM, params, True, 0.005, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.steerLimitUnwind
  assert not adaptive_log.freezeReason & GuardedResponseReason.STEER_LIMITED
  assert not adaptive_log.blockReason & GuardedResponseReason.STEER_LIMITED
  assert adaptive_log.sampleRejectReason & EstimatorRejectReason.STEER_LIMITED


def test_v3_native_torque_controller_logs_model_state():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, make_pose(), False, 0.2)

  assert lac_log.version == 3
  assert lac_log.adaptiveTorqueState.modelMode in (TorqueModelMode.native, TorqueModelMode.learned)
  assert lac_log.adaptiveTorqueState.authorityScale > 0.0


def test_v3_attenuates_nominal_output_before_assist(monkeypatch):
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = -12.0
  CS.steeringRateDeg = 0.0
  params = log.LiveParametersData.new_message()
  raw_measurement = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll) * CS.vEgo**2
  desired_curvature = (raw_measurement - 0.3) / CS.vEgo**2
  captured = {}

  def fake_attenuate(nominal_torque, desired_lateral_accel, actual_lateral_accel):
    captured["nominal_torque"] = nominal_torque
    captured["desired_lateral_accel"] = desired_lateral_accel
    captured["actual_lateral_accel"] = actual_lateral_accel
    captured["attenuated_torque"] = nominal_torque * 0.25
    return captured["attenuated_torque"]

  monkeypatch.setattr(latcontrol_torque_v3, "attenuate_same_direction_over_response", fake_attenuate, raising=False)

  _, _, lac_log = controller.update(True, CS, VM, params, False, desired_curvature, make_pose(), False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert captured["actual_lateral_accel"] > captured["desired_lateral_accel"] + 0.12
  assert captured["nominal_torque"] > 0.0
  assert adaptive_log.nominalOutput == pytest.approx(-captured["attenuated_torque"])


def test_v3_fallback_preserves_attenuated_nominal_output(monkeypatch):
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = -12.0
  CS.steeringRateDeg = 0.0
  params = log.LiveParametersData.new_message()
  raw_measurement = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll) * CS.vEgo**2
  desired_curvature = (raw_measurement - 0.3) / CS.vEgo**2
  captured = {}

  def fake_attenuate(nominal_torque, desired_lateral_accel, actual_lateral_accel):
    captured["nominal_torque"] = nominal_torque
    captured["desired_lateral_accel"] = desired_lateral_accel
    captured["actual_lateral_accel"] = actual_lateral_accel
    captured["attenuated_torque"] = nominal_torque * 0.25
    return captured["attenuated_torque"]

  def reject_update(_observation):
    return EstimatorResult(
      params=controller.estimator.state.params,
      confidence=0.96,
      positive_coverage=0.7,
      negative_coverage=0.7,
      residual_error=0.0,
      response_delay=0.2,
      sample_accepted=False,
      reject_reason=EstimatorRejectReason.RESIDUAL_SPIKE,
    )

  monkeypatch.setattr(latcontrol_torque_v3, "attenuate_same_direction_over_response", fake_attenuate, raising=False)
  controller.estimator.update = reject_update

  _, _, lac_log = controller.update(True, CS, VM, params, False, desired_curvature, make_pose(), False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.fallbackActive
  assert adaptive_log.authorityBand == AuthorityBand.limited
  assert adaptive_log.nominalOutput == pytest.approx(-captured["attenuated_torque"])
  assert abs(lac_log.output) < abs(captured["nominal_torque"])
  assert abs(lac_log.output) <= abs(captured["attenuated_torque"]) + 1e-6


def test_v3_native_torque_starts_near_full_authority():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()

  assert np.isclose(controller.estimator.state.confidence, 0.8)

  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.0, make_pose(), False, 0.2)

  assert lac_log.adaptiveTorqueState.modelMode == TorqueModelMode.native
  assert lac_log.adaptiveTorqueState.authorityBand == AuthorityBand.near_full
  assert np.isclose(lac_log.adaptiveTorqueState.authorityScale, 0.85)


def test_v3_native_faulting_frame_demotes_authority_telemetry():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringAngleDeg = 5.0
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, make_pose(), False, 0.2)

  assert lac_log.adaptiveTorqueState.sampleRejectReason & EstimatorRejectReason.SIGN_CONFLICT
  assert lac_log.adaptiveTorqueState.authorityBand == AuthorityBand.limited
  assert np.isclose(lac_log.adaptiveTorqueState.authorityScale, 0.45)
  assert lac_log.adaptiveTorqueState.fallbackActive
  assert abs(lac_log.output) <= 0.45 + 1e-6


def test_v3_sign_conflict_shaping_does_not_create_estimator_saturation():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()
  captured_observation = None

  def safety_update(inputs):
    shaping_result = ConservativeOutputShaperResult(
      inputs.unshaped_output,
      True,
      ConservativeOutputShapingReason.SIGN_CONFLICT,
      1.0,
      inputs.unshaped_output,
      0.8,
    )
    return TorqueV3SafetyResult(inputs.unshaped_output, False, 1.0, shaping_result)

  def estimator_update(observation):
    nonlocal captured_observation
    captured_observation = observation
    return EstimatorResult(
      params=controller.estimator.state.params,
      confidence=0.8,
      positive_coverage=0.0,
      negative_coverage=0.0,
      residual_error=0.0,
      response_delay=0.2,
      sample_accepted=False,
      reject_reason=EstimatorRejectReason.SIGN_CONFLICT | EstimatorRejectReason.STEER_LIMITED,
    )

  controller.safety_envelope.update = safety_update
  controller.estimator.update = estimator_update

  controller.update(True, CS, VM, params, True, 0.001, make_pose(), False, 0.2)

  assert captured_observation is not None
  assert captured_observation.steer_limited_by_safety
  assert not captured_observation.saturated


def test_v3_native_low_command_frames_keep_near_full_authority():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()

  for _ in range(10):
    _, _, lac_log = controller.update(True, CS, VM, params, False, 0.0, make_pose(), False, 0.2)

    assert lac_log.adaptiveTorqueState.sampleRejectReason & EstimatorRejectReason.LOW_COMMAND

  assert lac_log.adaptiveTorqueState.modelMode == TorqueModelMode.native
  assert lac_log.adaptiveTorqueState.authorityBand == AuthorityBand.near_full
  assert np.isclose(lac_log.adaptiveTorqueState.authorityScale, 0.85)


def test_v3_residual_and_stale_fault_frames_cap_output_to_limited_authority():
  for reject_reason in (EstimatorRejectReason.RESIDUAL_SPIKE, EstimatorRejectReason.STALE_MODEL):
    controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
    CS = car.CarState.new_message()
    CS.vEgo = 30.0
    params = log.LiveParametersData.new_message()
    estimator_params = controller.estimator.state.params

    def reject_update(_observation, reject_reason=reject_reason, estimator_params=estimator_params):
      return EstimatorResult(
        params=estimator_params,
        confidence=0.96,
        positive_coverage=0.7,
        negative_coverage=0.7,
        residual_error=0.0,
        response_delay=0.2,
        sample_accepted=False,
        reject_reason=reject_reason,
      )

    controller.estimator.state.confidence = 0.96
    controller.estimator.state.positive_coverage = 0.7
    controller.estimator.state.negative_coverage = 0.7
    controller.estimator.update = reject_update

    _, _, lac_log = controller.update(True, CS, VM, params, False, 0.003, make_pose(), False, 0.2)

    assert lac_log.adaptiveTorqueState.sampleRejectReason & reject_reason
    assert lac_log.adaptiveTorqueState.authorityBand == AuthorityBand.limited
    assert np.isclose(lac_log.adaptiveTorqueState.authorityScale, 0.45)
    assert lac_log.adaptiveTorqueState.fallbackActive
    assert abs(lac_log.output) <= 0.45 + 1e-6


def test_v3_alternating_fault_and_clean_estimator_frames_do_not_restore_full_output():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 30.0
  params = log.LiveParametersData.new_message()
  estimator_params = controller.estimator.state.params
  update_count = 0

  def alternating_update(observation):
    del observation
    nonlocal update_count
    update_count += 1
    faulting = update_count % 2 == 1
    return EstimatorResult(
      params=estimator_params,
      confidence=0.0 if faulting else 0.96,
      positive_coverage=0.7,
      negative_coverage=0.7,
      residual_error=0.0,
      response_delay=0.2,
      sample_accepted=not faulting,
      reject_reason=EstimatorRejectReason.RESIDUAL_SPIKE if faulting else EstimatorRejectReason.NONE,
    )

  controller.estimator.update = alternating_update

  for _ in range(24):
    _, _, lac_log = controller.update(True, CS, VM, params, False, 0.003, make_pose(), False, 0.2)

    assert lac_log.adaptiveTorqueState.authorityBand == AuthorityBand.limited
    assert np.isclose(lac_log.adaptiveTorqueState.authorityScale, 0.45)
    assert lac_log.adaptiveTorqueState.fallbackActive
    assert abs(lac_log.output) <= 0.45 + 1e-6


def test_v3_synthetic_pid_origin_starts_limited():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2, force_pid=True)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, make_pose(), False, 0.2)

  assert lac_log.version == 3
  assert lac_log.adaptiveTorqueState.modelMode == TorqueModelMode.synthetic
  assert lac_log.adaptiveTorqueState.authorityBand == AuthorityBand.limited
  assert np.isclose(lac_log.adaptiveTorqueState.authorityScale, 0.45)


def test_v3_one_sided_synthetic_learning_does_not_activate_learned_mode():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2, force_pid=True)
  controller.estimator.state.confidence = 0.96
  controller.estimator.state.positive_coverage = 0.9
  controller.estimator.state.negative_coverage = 0.0
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()

  for _ in range(3):
    _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, make_pose(), False, 0.2)

    assert lac_log.adaptiveTorqueState.sampleAccepted
    assert lac_log.adaptiveTorqueState.modelMode == TorqueModelMode.synthetic
    assert lac_log.adaptiveTorqueState.authorityBand != AuthorityBand.full


def test_v3_learned_activation_refreshes_active_torque_params_and_pid_limits():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2, force_pid=True)
  controller.estimator.state.params.lat_accel_factor = 1.0
  controller.estimator.state.confidence = 0.96
  controller.estimator.state.positive_coverage = 0.6
  controller.estimator.state.negative_coverage = 0.6
  previous_pos_limit = controller.pid.pos_limit
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, make_pose(), False, 0.2)

  assert lac_log.adaptiveTorqueState.modelMode == TorqueModelMode.learned
  assert controller.torque_params is controller.model_adapter.params
  assert controller.extension.torque_params is controller.torque_params
  assert controller.torque_params.latAccelFactor != 2.5
  assert controller.pid.pos_limit != previous_pos_limit
  assert np.isclose(controller.pid.pos_limit, controller.lateral_accel_from_torque(controller.steer_max, controller.torque_params))
  assert np.isclose(controller.pid.neg_limit, controller.lateral_accel_from_torque(-controller.steer_max, controller.torque_params))


def test_v3_live_torque_update_does_not_mutate_learned_params():
  controller, _ = get_controller(TOYOTA.TOYOTA_RAV4)
  learned_params = TorqueModelParams(1.1, 0.2, 0.05)
  assert controller.model_adapter.update_learned_params(learned_params, 0.96)
  controller.torque_params = controller.model_adapter.params
  controller.extension.torque_params = controller.torque_params
  controller.update_limits()
  previous_pos_limit = controller.pid.pos_limit

  controller.update_live_torque_params(4.0, 0.9, 0.8)

  assert controller.model_adapter.mode == TorqueModelMode.learned
  assert controller.torque_params is controller.model_adapter.params
  assert controller.extension.torque_params is controller.torque_params
  assert controller.torque_params.latAccelFactor == pytest.approx(1.1)
  assert controller.torque_params.latAccelOffset == pytest.approx(0.2)
  assert controller.torque_params.friction == pytest.approx(0.05)
  assert controller.pid.pos_limit == pytest.approx(previous_pos_limit)


def test_v3_direct_extension_model_update_resets_model_age_for_estimator():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2, force_pid=True)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()
  model_v2 = make_flat_model_v2()

  for _ in range(30):
    controller.extension.update_model_v2(model_v2)
    _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, make_pose(), False, 0.2)

  assert not lac_log.adaptiveTorqueState.sampleRejectReason & EstimatorRejectReason.STALE_MODEL


def test_v3_authority_limited_output_counts_as_controller_saturation():
  controller, VM = get_controller(TOYOTA.TOYOTA_RAV4)
  CS = car.CarState.new_message()
  CS.vEgo = 30.0
  params = log.LiveParametersData.new_message()

  for _ in range(1000):
    _, _, lac_log = controller.update(True, CS, VM, params, False, 1.0, make_pose(), False, 0.2)

  assert lac_log.adaptiveTorqueState.authorityScale < 1.0
  assert lac_log.saturated


def test_v3_authority_limited_frame_rejects_estimator_sample():
  controller, VM = get_controller(TOYOTA.TOYOTA_RAV4)
  CS = car.CarState.new_message()
  CS.vEgo = 30.0
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 1.0, make_pose(), False, 0.2)

  assert lac_log.adaptiveTorqueState.authorityScale < 1.0
  assert not lac_log.adaptiveTorqueState.sampleAccepted
  assert lac_log.adaptiveTorqueState.sampleRejectReason & EstimatorRejectReason.SATURATED


def test_v3_smoke_on_gm_nonlinear_native_torque_platform():
  controller, VM = get_controller(GM.CHEVROLET_BOLT_EUV)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringAngleDeg = 5.0
  CS.steeringRateDeg = 25.0
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, make_pose(), False, 0.2)

  assert lac_log.version == 3
  assert np.isfinite(lac_log.output)
