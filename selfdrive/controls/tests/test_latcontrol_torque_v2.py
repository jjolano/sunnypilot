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
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import adjust_future_time_for_longitudinal_accel
from openpilot.sunnypilot.selfdrive.controls.lib.torque_conservative_output_shaper import ConservativeOutputShaperResult, ConservativeOutputShapingReason
from openpilot.sunnypilot.selfdrive.controls.lib.torque_disturbance import TorqueDisturbanceReason, TorqueDisturbanceState
from openpilot.sunnypilot.selfdrive.controls.lib.torque_guarded_response_assist import GuardedResponseReason
from openpilot.sunnypilot.selfdrive.controls.lib.steering_actuator_feedback import SteeringActuatorFeedback, SteeringLimitReason
from openpilot.sunnypilot.selfdrive.locationd.speed_aware_torque import SPEED_AWARE_PARAMS_VERSION

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
LatControlTorqueV21 = latcontrol_torque_v2.LatControlTorqueV21
RefinedOutputGovernorInputs = latcontrol_torque_v2.RefinedOutputGovernorInputs
RefinedOutputGovernorReason = latcontrol_torque_v2.RefinedOutputGovernorReason
TorqueV21RefinedOutputGovernor = latcontrol_torque_v2.TorqueV21RefinedOutputGovernor


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


def get_v21_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CP_SP.neuralNetworkLateralControl.model.path = MOCK_MODEL_PATH
  CI = CarInterface(CP, CP_SP)
  CP_SP = convert_to_capnp(CP_SP)
  VM = VehicleModel(CP)
  controller = LatControlTorqueV21(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)
  return controller, VM


def make_governor_inputs(**overrides):
  values = {
    "active": True,
    "v_ego": 20.0,
    "steering_pressed": False,
    "steering_rate_deg": 0.0,
    "same_direction_limit": False,
    "output_torque": 1.0,
    "max_output": 1.0,
    "desired_lateral_accel": 0.0,
    "actual_lateral_accel": 0.0,
  }
  values.update(overrides)
  return RefinedOutputGovernorInputs(**values)


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


class CapturingNNTorqueModel:
  friction_override = False

  def __init__(self):
    self.inputs = []

  def evaluate(self, input_array):
    self.inputs.append(list(input_array))
    return 0.0


def enable_flat_nnlc(controller):
  controller.extension.enabled = True
  controller.extension.has_nn_model = True
  controller.extension.model = FlatNNTorqueModel()
  controller.extension.update_model_v2(make_flat_model_v2())


def enable_capturing_nnlc(controller, model_v2, hardening=False):
  capturing_model = CapturingNNTorqueModel()
  controller.extension.enabled = True
  controller.extension.has_nn_model = True
  controller.extension.model = capturing_model
  controller.extension.control_calculation_hardening = hardening
  controller.extension.update_lateral_lag(0.2)
  controller.extension.update_model_v2(model_v2)
  return capturing_model


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


@pytest.mark.parametrize("acceleration_y", [
  [],
  [0.0, 0.0, 0.0, 0.0],
  [float("nan")] + [0.0 for _ in ModelConstants.T_IDXS[1:]],
])
def test_nnlc_model_invalid_without_valid_lateral_acceleration_plan(acceleration_y):
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  model_v2 = make_flat_model_v2()
  model_v2.acceleration.y = acceleration_y

  controller.extension.update_model_v2(model_v2)

  assert not controller.extension.model_valid


@pytest.mark.parametrize("orientation_y", [
  [],
  [0.0, 0.0, 0.0, 0.0],
  [float("nan")] + [0.0 for _ in ModelConstants.T_IDXS[1:]],
])
def test_nnlc_model_invalid_without_valid_orientation_y_plan(orientation_y):
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  model_v2 = make_flat_model_v2()
  model_v2.orientation.y = orientation_y

  controller.extension.update_model_v2(model_v2)

  assert not controller.extension.model_valid


def test_nnlc_future_time_defaults_to_kinematic_accel_adjustment():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  model_v2 = make_flat_model_v2()
  model_v2.acceleration.y = [float(t) for t in ModelConstants.T_IDXS]
  capturing_model = enable_capturing_nnlc(controller, model_v2, hardening=False)
  CS = car.CarState.new_message()
  CS.vEgo = 10.0
  CS.aEgo = 2.0
  params = log.LiveParametersData.new_message()

  controller.update(True, CS, VM, params, False, 0.001, make_pose(), False, 0.2)

  expected_future_times = [adjust_future_time_for_longitudinal_accel(t, CS.vEgo, CS.aEgo)
                           for t in controller.extension.nn_future_times]
  assert capturing_model.inputs[-1][7:11] == pytest.approx(expected_future_times)


def test_nnlc_hardening_uses_kinematic_accel_adjustment():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  model_v2 = make_flat_model_v2()
  model_v2.acceleration.y = [float(t) for t in ModelConstants.T_IDXS]
  capturing_model = enable_capturing_nnlc(controller, model_v2, hardening=True)
  CS = car.CarState.new_message()
  CS.vEgo = 10.0
  CS.aEgo = 2.0
  params = log.LiveParametersData.new_message()

  controller.update(True, CS, VM, params, False, 0.001, make_pose(), False, 0.2)

  expected_future_times = [adjust_future_time_for_longitudinal_accel(t, CS.vEgo, CS.aEgo)
                           for t in controller.extension.nn_future_times]
  assert capturing_model.inputs[-1][7:11] == pytest.approx(expected_future_times)


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


def test_v21_refined_governor_slew_limits_when_no_under_response():
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)

  result = governor.update(make_governor_inputs(v_ego=20.0, output_torque=1.0))

  assert result.reason & RefinedOutputGovernorReason.SLEW_LIMITED
  assert 0.0 < result.output_torque < 1.0


def test_v21_under_response_floor_bypasses_slew_below_nine_mps():
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)

  result = governor.update(make_governor_inputs(v_ego=5.0, output_torque=1.0, desired_lateral_accel=0.5, actual_lateral_accel=0.0))

  assert result.reason & RefinedOutputGovernorReason.UNDER_RESPONSE_FLOOR
  assert result.output_torque == pytest.approx(1.0)
  assert not result.reason & RefinedOutputGovernorReason.SLEW_LIMITED


def test_v21_under_response_floor_fades_between_nine_and_twelve_mps():
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)

  faded = governor.update(make_governor_inputs(v_ego=10.5, output_torque=1.0, desired_lateral_accel=0.5, actual_lateral_accel=0.0))
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)
  unprotected = governor.update(make_governor_inputs(v_ego=13.0, output_torque=1.0, desired_lateral_accel=0.5, actual_lateral_accel=0.0))

  assert faded.reason & RefinedOutputGovernorReason.UNDER_RESPONSE_FLOOR
  assert unprotected.reason & RefinedOutputGovernorReason.SLEW_LIMITED
  assert unprotected.output_torque < faded.output_torque < 1.0


def test_v21_under_response_floor_protects_low_speed_corrective_reversal():
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)
  governor.previous_output = 0.7

  result = governor.update(make_governor_inputs(v_ego=5.0, output_torque=-1.0, desired_lateral_accel=-0.5, actual_lateral_accel=0.2))

  assert result.reason & RefinedOutputGovernorReason.SIGN_CHANGE_LIMITED
  assert result.reason & RefinedOutputGovernorReason.UNDER_RESPONSE_FLOOR
  assert result.output_torque == pytest.approx(-1.0)


def test_v21_same_direction_limit_uses_soft_cap_without_floor():
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)
  governor.previous_output = 0.9

  result = governor.update(make_governor_inputs(same_direction_limit=True, output_torque=1.0))

  assert result.reason & RefinedOutputGovernorReason.SAME_DIRECTION_LIMIT
  assert result.reason & RefinedOutputGovernorReason.CLIPPED
  assert result.output_torque == pytest.approx(0.85)


def test_v21_same_direction_limit_recovers_faster_at_high_speed():
  low_speed_governor = TorqueV21RefinedOutputGovernor(DT_CTRL)
  low_speed_governor.previous_output = 0.2
  high_speed_governor = TorqueV21RefinedOutputGovernor(DT_CTRL)
  high_speed_governor.previous_output = 0.2

  low_speed = low_speed_governor.update(make_governor_inputs(v_ego=8.0, same_direction_limit=True, output_torque=1.0))
  high_speed = high_speed_governor.update(make_governor_inputs(v_ego=28.0, same_direction_limit=True, output_torque=1.0))

  assert low_speed.reason & RefinedOutputGovernorReason.SAME_DIRECTION_LIMIT
  assert high_speed.reason & RefinedOutputGovernorReason.SAME_DIRECTION_LIMIT
  assert low_speed.reason & RefinedOutputGovernorReason.SLEW_LIMITED
  assert high_speed.reason & RefinedOutputGovernorReason.SLEW_LIMITED
  assert high_speed.output_torque > low_speed.output_torque


def test_v21_high_rate_soft_cap_applies_outside_under_response_floor():
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)
  governor.previous_output = 1.0

  result = governor.update(make_governor_inputs(steering_rate_deg=100.0, output_torque=1.0))

  assert result.reason & RefinedOutputGovernorReason.HIGH_STEERING_RATE
  assert result.reason & RefinedOutputGovernorReason.CLIPPED
  assert result.output_torque == pytest.approx(0.62)


def test_v21_driver_override_uses_fast_bounded_release():
  governor = TorqueV21RefinedOutputGovernor(DT_CTRL)
  governor.previous_output = 1.0

  result = governor.update(make_governor_inputs(steering_pressed=True, output_torque=1.0))

  assert result.reason & RefinedOutputGovernorReason.DRIVER_OVERRIDE
  assert result.output_torque == pytest.approx(0.94)


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
  assert adaptive_log.governorReason == 0
  assert abs(adaptive_log.unshapedOutput - (adaptive_log.nominalOutput + adaptive_log.assistOutput + adaptive_log.biasOutput)) < 1e-6
  assert adaptive_log.unshapedOutput == lac_log.output


def test_v21_logs_version_and_separate_governor_reason():
  controller, VM = get_v21_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, make_pose(), False, 0.2)
  adaptive_log = lac_log.adaptiveTorqueState

  assert lac_log.version == 21
  assert adaptive_log.governorReason & RefinedOutputGovernorReason.SLEW_LIMITED
  assert adaptive_log.shapingReason == 0
  assert abs(lac_log.output) < abs(adaptive_log.unshapedOutput)


def test_v2_softens_low_demand_friction_driven_reversals():
  controller, VM = get_controller(TOYOTA.TOYOTA_RAV4_TSS2)

  CS = car.CarState.new_message()
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()
  pose = make_pose()
  logs = []
  previous_angle = 0.0

  for i in range(400):
    t = i * DT_CTRL
    CS.vEgo = 16.0 - 0.5 * t / 4.0
    desired_lateral_accel = 0.08 * math.sin(2.0 * math.pi * 0.58 * t + 2.2) + 0.03 * math.sin(2.0 * math.pi * 1.1 * t)
    steering_angle = 0.7 + 0.85 * math.sin(2.0 * math.pi * 0.65 * t) + 0.25 * math.sin(2.0 * math.pi * 1.4 * t + 1.0)
    CS.steeringAngleDeg = steering_angle
    CS.steeringRateDeg = (steering_angle - previous_angle) / DT_CTRL if i > 0 else 0.0
    previous_angle = steering_angle

    _, _, lac_log = controller.update(True, CS, VM, params, False, desired_lateral_accel / CS.vEgo**2, pose, False, 0.2)
    logs.append(lac_log)

  settled_logs = logs[50:]
  desired_lateral_accel = np.array([lac_log.desiredLateralAccel for lac_log in settled_logs])
  actual_lateral_accel = np.array([lac_log.actualLateralAccel for lac_log in settled_logs])
  feedforward = np.array([lac_log.f for lac_log in settled_logs])
  unshaped_output = np.array([lac_log.adaptiveTorqueState.unshapedOutput for lac_log in settled_logs])

  assert max(np.max(np.abs(desired_lateral_accel)), np.max(np.abs(actual_lateral_accel))) < 0.2
  assert np.ptp(feedforward) < 0.25
  assert np.ptp(unshaped_output) < 0.45


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


class SpeedAdaptiveApplyParams:
  def __init__(self, enabled: bool):
    self.enabled = enabled

  def get_bool(self, key: str) -> bool:
    return self.enabled if key == "LiveTorqueSpeedAdaptiveApplyToggle" else False


def make_speed_adaptive_payload(controller, buckets, **overrides):
  CP = controller.extension.CP
  payload = {
    "version": SPEED_AWARE_PARAMS_VERSION,
    "carFingerprint": CP.carFingerprint,
    "lateralTuning": CP.lateralTuning.which(),
    "torqueLatAccelFactor": float(CP.lateralTuning.torque.latAccelFactor),
    "torqueFriction": float(CP.lateralTuning.torque.friction),
    "buckets": buckets,
  }
  payload.update(overrides)
  return payload


def apply_speed_adaptive_factor(controller, factor):
  controller.extension.last_v_ego = 5.0
  controller.extension.speed_adaptive_apply_enabled = True
  controller.extension.speed_aware_params = {"0_10": (factor, 0.0, 0.0)}
  assert controller.extension.update_override_torque_params(controller.torque_params)


def test_v2_speed_adaptive_factor_restores_base_when_disabled():
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  base_factor = controller.torque_params.latAccelFactor
  adaptive_factor = base_factor * 1.25

  apply_speed_adaptive_factor(controller, adaptive_factor)
  controller.extension.speed_adaptive_apply_enabled = False
  restored = controller.extension.update_override_torque_params(controller.torque_params)

  assert restored
  assert controller.torque_params.latAccelFactor == pytest.approx(base_factor)


def test_v2_speed_adaptive_factor_restores_base_when_params_invalid():
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  base_factor = controller.torque_params.latAccelFactor
  adaptive_factor = base_factor * 1.25

  apply_speed_adaptive_factor(controller, adaptive_factor)
  controller.extension.speed_aware_params = {"0_10": (base_factor * 3.0, 0.0, 0.0)}
  restored = controller.extension.update_override_torque_params(controller.torque_params)

  assert restored
  assert controller.torque_params.latAccelFactor == pytest.approx(base_factor)


def test_speed_adaptive_params_reject_unversioned_payload():
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  controller.extension.params = SpeedAdaptiveApplyParams(True)
  base_factor = controller.torque_params.latAccelFactor

  controller.extension.update_speed_aware_params(str({"0_10": (base_factor * 1.1, 0.0, 0.0)}))

  assert controller.extension.speed_aware_params is None


def test_speed_adaptive_params_reject_wrong_car_payload():
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  controller.extension.params = SpeedAdaptiveApplyParams(True)
  base_factor = controller.torque_params.latAccelFactor

  payload = make_speed_adaptive_payload(controller, {"0_10": (base_factor * 1.1, 0.0, 0.0)}, carFingerprint="different-car")
  controller.extension.update_speed_aware_params(str(payload))

  assert controller.extension.speed_aware_params is None


def test_speed_adaptive_params_reject_wrong_torque_tune_payload():
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  controller.extension.params = SpeedAdaptiveApplyParams(True)
  base_factor = controller.torque_params.latAccelFactor

  payload = make_speed_adaptive_payload(controller, {"0_10": (base_factor * 1.1, 0.0, 0.0)}, torqueLatAccelFactor=base_factor * 1.5)
  controller.extension.update_speed_aware_params(str(payload))

  assert controller.extension.speed_aware_params is None


@pytest.mark.parametrize("bucket_value", [[], "bad", (math.nan, 0.0, 0.0), (1.0, 0.0)])
def test_speed_adaptive_params_reject_malformed_bucket_values(bucket_value):
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  controller.extension.params = SpeedAdaptiveApplyParams(True)

  payload = make_speed_adaptive_payload(controller, {"0_10": bucket_value})
  controller.extension.update_speed_aware_params(str(payload))

  assert controller.extension.speed_aware_params is None


def test_speed_adaptive_params_accept_matching_metadata_payload():
  controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  controller.extension.params = SpeedAdaptiveApplyParams(True)
  base_factor = controller.torque_params.latAccelFactor
  buckets = {"0_10": (base_factor * 1.1, 0.0, 0.0)}

  controller.extension.update_speed_aware_params(str(make_speed_adaptive_payload(controller, buckets)))

  assert controller.extension.speed_aware_params == buckets
