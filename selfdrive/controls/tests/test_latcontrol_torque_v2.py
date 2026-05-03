import math
import numpy as np
import pytest
import sys
import types

from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.gm.values import CAR as GM
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.locationd.helpers import Measurement, Pose
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import ConservativeOutputShaperResult, ConservativeOutputShapingReason
from openpilot.sunnypilot.selfdrive.controls.lib.torque_disturbance import TorqueDisturbanceReason, TorqueDisturbanceState
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

from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_v2

LatControlTorque = latcontrol_torque_v2.LatControlTorque


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CP_SP.neuralNetworkLateralControl.model.path = MOCK_MODEL_PATH
  CI = CarInterface(CP, CP_SP)
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


def test_v2_uses_crawl_speed_for_low_speed_pid_gain():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 1.0
  params = log.LiveParametersData.new_message()

  controller.update(True, CS, VM, params, False, 0.0, make_pose(), False, 0.2)

  assert controller.pid.speed == pytest.approx(3.0)


def test_v2_nnlc_uses_crawl_speed_for_low_speed_pid_gain():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  enable_flat_nnlc(controller)
  CS = car.CarState.new_message()
  CS.vEgo = 1.0
  params = log.LiveParametersData.new_message()

  controller.update(True, CS, VM, params, False, 0.0, make_pose(), False, 0.2)

  assert controller.extension._pid.speed == pytest.approx(3.0)


def test_measurement_smoother_predicts_between_held_angle_updates():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  first = smoother.update(True, 20.0, False, 0.0, -1.0)
  second = smoother.update(True, 20.0, False, 0.0, -1.0)

  assert first == 0.0
  assert second < -0.005
  assert second > -0.02


def test_measurement_smoother_softens_raw_angle_jump_after_hold():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  first = smoother.update(True, 20.0, False, 0.0, -1.0)
  held = smoother.update(True, 20.0, False, 0.0, -1.0)
  jumped = smoother.update(True, 20.0, False, -0.02, -1.0)

  assert held < first
  assert jumped < held
  assert abs(jumped - held) < 0.02


def test_measurement_smoother_resets_on_override():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, -1.0)
  smoother.update(True, 20.0, False, 0.0, -1.0)

  reset_value = smoother.update(True, 20.0, True, 0.5, -10.0)
  resumed_value = smoother.update(True, 20.0, False, 0.5, -10.0)

  assert reset_value == 0.5
  assert resumed_value == 0.5


def test_measurement_smoother_resets_when_inactive():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, -1.0)
  smoother.update(True, 20.0, False, 0.0, -1.0)

  reset_value = smoother.update(False, 20.0, False, 0.5, -10.0)
  resumed_value = smoother.update(True, 20.0, False, 0.5, -10.0)

  assert reset_value == 0.5
  assert resumed_value == 0.5


def test_measurement_smoother_resets_at_low_speed():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, -1.0)
  smoother.update(True, 20.0, False, 0.0, -1.0)

  reset_value = smoother.update(True, 4.0, False, 0.5, -10.0)
  resumed_value = smoother.update(True, 20.0, False, 0.5, -10.0)

  assert reset_value == 0.5
  assert resumed_value == 0.5


def test_measurement_smoother_resets_on_non_finite_raw_measurement():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, 1.0)
  reset_value = smoother.update(True, 20.0, False, float("nan"), 1.0)
  resumed_value = smoother.update(True, 20.0, False, 0.5, 1.0)

  assert reset_value == 0.0
  assert resumed_value == 0.5


def test_measurement_smoother_clamps_prediction_from_large_rate():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, 20.0)
  predicted = smoother.update(True, 20.0, False, 0.0, 20.0)

  assert 0.0 < predicted <= 0.04


def test_measurement_smoother_resets_after_implausible_rate_spike():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, 1.0)
  reset_value = smoother.update(True, 20.0, False, 0.5, 500.0)
  resumed_value = smoother.update(True, 20.0, False, 0.5, 1.0)

  assert reset_value == 0.5
  assert resumed_value == 0.5


def test_measurement_smoother_resets_on_non_finite_rate():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, 1.0)
  reset_value = smoother.update(True, 20.0, False, 0.5, float("nan"))
  resumed_value = smoother.update(True, 20.0, False, 0.5, 1.0)

  assert reset_value == 0.5
  assert resumed_value == 0.5


def test_measurement_smoother_limits_lag_from_raw_measurement():
  smoother = latcontrol_torque_v2.LateralAccelMeasurementSmoother(DT_CTRL)

  smoother.update(True, 20.0, False, 0.0, 0.0)
  jumped = smoother.update(True, 20.0, False, 3.0, 0.0)

  assert 2.0 <= jumped < 3.0


def test_v2_conditions_measurement_between_held_angle_updates():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = 10.0
  CS.steeringRateDeg = 20.0
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  _, _, first_log = controller.update(True, CS, VM, params, False, 0.0, pose, False, 0.2)
  _, _, second_log = controller.update(True, CS, VM, params, False, 0.0, pose, False, 0.2)

  assert second_log.actualLateralAccel < first_log.actualLateralAccel - 0.001


def test_v2_measurement_smoother_smoke_on_non_toyota_torque_platform():
  controller, VM = get_controller(GM.CHEVROLET_BOLT_EUV)

  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = 5.0
  CS.steeringRateDeg = 25.0
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  lac_log = None
  for _ in range(3):
    _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, pose, False, 0.2)

  assert lac_log is not None
  assert lac_log.version == 2
  assert np.isfinite(lac_log.actualLateralAccel)


def test_v2_resets_measurement_rate_filter_on_smoother_reset():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  controller.update(True, CS, VM, params, False, 0.0, pose, False, 0.2)

  CS.steeringAngleDeg = -14.0
  CS.steeringPressed = True
  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.0, pose, False, 0.2)

  assert controller.measurement_rate_filter.x == 0.0
  assert abs(controller.previous_measurement - lac_log.actualLateralAccel) < 1e-6


def test_v2_logging_fields_are_populated():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 20
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  lac_log = None
  for _ in range(80):
    _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, pose, False, 0.2)

  assert lac_log is not None
  assert lac_log.version == 2
  adaptive_log = lac_log.adaptiveTorqueState
  assert adaptive_log.active
  assert adaptive_log.nominalOutput != 0.0
  assert adaptive_log.freezeReason == 0
  assert not adaptive_log.shapingActive
  assert adaptive_log.shapingReason == 0
  assert adaptive_log.shapingConfidence == 0.0
  assert adaptive_log.outputCap == 1.0
  assert abs(adaptive_log.unshapedOutput - (adaptive_log.nominalOutput + adaptive_log.assistOutput + adaptive_log.biasOutput)) < 1e-6
  assert adaptive_log.unshapedOutput == lac_log.output


def test_v2_attenuates_nominal_output_before_assist(monkeypatch):
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

  monkeypatch.setattr(latcontrol_torque_v2, "attenuate_same_direction_over_response", fake_attenuate, raising=False)

  _, _, lac_log = controller.update(True, CS, VM, params, False, desired_curvature, make_pose(), False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert captured["actual_lateral_accel"] > captured["desired_lateral_accel"] + 0.12
  assert captured["nominal_torque"] > 0.0
  assert adaptive_log.nominalOutput == pytest.approx(-captured["attenuated_torque"])


def test_v2_disturbance_telemetry_clean_when_tracking_cleanly():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.0, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.disturbanceState == TorqueDisturbanceState.NONE
  assert adaptive_log.disturbanceReason == TorqueDisturbanceReason.NONE
  assert adaptive_log.disturbanceConfidence == 0.0


def test_v2_release_on_override():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 6
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  for _ in range(40):
    controller.update(True, CS, VM, params, False, 2e-4, pose, False, 0.2)

  CS.steeringPressed = True
  _, _, lac_log = controller.update(True, CS, VM, params, False, 2e-5, pose, False, 0.2)
  assert lac_log.adaptiveTorqueState.releaseActive
  assert lac_log.adaptiveTorqueState.phase == log.ControlsState.LateralTorqueState.AdaptiveTorqueState.Phase.release
  assert lac_log.adaptiveTorqueState.freezeReason != 0
  assert lac_log.adaptiveTorqueState.blockReason != 0
  assert lac_log.adaptiveTorqueState.shapingActive
  assert lac_log.adaptiveTorqueState.shapingReason & ConservativeOutputShapingReason.STEERING_PRESSED
  assert lac_log.adaptiveTorqueState.shapingReason & ConservativeOutputShapingReason.RELEASE
  assert abs(lac_log.adaptiveTorqueState.outputCap - 0.8) < 1e-6
  assert abs(lac_log.output) <= abs(lac_log.adaptiveTorqueState.unshapedOutput)


def test_v2_reports_learning_frozen_at_low_speed():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 3.5
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  lac_log = None
  for _ in range(20):
    _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, pose, False, 0.2)

  assert lac_log is not None
  assert lac_log.version == 2
  assert lac_log.adaptiveTorqueState.learningFrozen
  assert lac_log.adaptiveTorqueState.freezeReason != 0


def test_v2_softens_low_speed_same_sign_unwind():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 5.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = -30.0
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  for _ in range(60):
    controller.update(True, CS, VM, params, False, 0.02, pose, False, 0.2)

  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.002, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert lac_log.error < 0.0
  assert lac_log.desiredLateralAccel < lac_log.actualLateralAccel
  assert adaptive_log.assistOutput <= 0.0
  assert adaptive_log.biasOutput < 0.0
  assert adaptive_log.shapingActive
  assert adaptive_log.shapingReason & ConservativeOutputShapingReason.SAME_SIGN_UNWIND
  assert abs(lac_log.output) < abs(adaptive_log.unshapedOutput)
  assert adaptive_log.nominalOutput < 0.95
  assert lac_log.output < 0.9
  assert abs(lac_log.output) <= abs(adaptive_log.unshapedOutput)


def test_v2_same_sign_unwind_release_uses_smoothed_measurement_sign():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 5.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = -30.0
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  for _ in range(60):
    controller.update(True, CS, VM, params, False, 0.02, pose, False, 0.2)

  CS.steeringAngleDeg = 10.0
  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.002, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert lac_log.desiredLateralAccel < lac_log.actualLateralAccel
  assert adaptive_log.blockReason & GuardedResponseReason.SAME_SIGN_UNWIND
  assert adaptive_log.shapingActive
  assert adaptive_log.shapingReason & ConservativeOutputShapingReason.SIGN_CONFLICT
  assert adaptive_log.shapingReason & ConservativeOutputShapingReason.SAME_SIGN_UNWIND
  assert abs(adaptive_log.outputCap - 0.3) < 1e-6
  assert abs(lac_log.output) < abs(adaptive_log.unshapedOutput) * 0.5


def test_v2_shapes_sign_conflict():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 15.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = 12.0
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.shapingActive
  assert adaptive_log.shapingReason & ConservativeOutputShapingReason.SIGN_CONFLICT
  assert abs(adaptive_log.outputCap - 0.8) < 1e-6
  assert abs(lac_log.output) <= abs(adaptive_log.unshapedOutput)
  assert adaptive_log.disturbanceState == TorqueDisturbanceState.ACTIVE
  assert adaptive_log.disturbanceReason & TorqueDisturbanceReason.SIGN_CONFLICT
  assert adaptive_log.disturbanceConfidence == 1.0


def test_v2_shapes_near_iso_accel_margin():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 30.0
  CS.steeringPressed = False
  CS.steeringAngleDeg = -14.0
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.004, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.shapingActive
  assert adaptive_log.shapingReason & ConservativeOutputShapingReason.NEAR_ISO_ACCEL
  assert adaptive_log.outputCap <= 0.9
  assert abs(lac_log.output) <= abs(adaptive_log.unshapedOutput)


def test_v2_shapes_near_iso_accel_when_raw_measurement_jumps_ahead_of_smoother():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 30.0
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  pose = make_pose()
  controller.update(True, CS, VM, params, False, 0.004, pose, False, 0.2)

  CS.steeringAngleDeg = -14.0
  CS.steeringRateDeg = 0.0
  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.004, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.shapingActive
  assert adaptive_log.shapingReason & ConservativeOutputShapingReason.NEAR_ISO_ACCEL
  assert adaptive_log.outputCap <= 0.9


def test_v2_bump_shaping_uses_raw_steering_rate_when_model_lookahead_is_zero():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 15.0
  CS.steeringPressed = False
  CS.steeringRateDeg = 300.0
  params = log.LiveParametersData.new_message()

  controller.extension.update_model_v2(make_flat_model_v2())

  pose = make_pose()
  _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, pose, False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert adaptive_log.shapingActive
  assert adaptive_log.shapingReason & ConservativeOutputShapingReason.BUMP
  assert adaptive_log.disturbanceState == TorqueDisturbanceState.ACTIVE
  assert adaptive_log.disturbanceReason & TorqueDisturbanceReason.BUMP_JERK
  assert adaptive_log.disturbanceConfidence > 0.0


def test_v2_signed_steer_limit_allows_clear_unwind_from_actuator_lag_cap():
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
  controller.output_shaper = spy

  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, 0.7, 0.45, 0.25, False, True)
  )
  controller.update(True, CS, VM, params, True, 0.005, pose, False, 0.2)

  assert spy.inputs is not None
  assert not spy.inputs.steer_limit_same_direction
  assert spy.inputs.steer_limit_unwind
  assert spy.inputs.steer_limit_requested_output == pytest.approx(-0.7)
  assert spy.inputs.steer_limit_applied_output == pytest.approx(-0.45)


def test_v2_signed_unwind_steer_limit_does_not_freeze_response_assist():
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


def test_v2_logs_signed_steer_limit_feedback():
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
