import sys
import types

import numpy as np
import pytest

from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.locationd.helpers import Measurement, Pose
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
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
V3GovernorReason = latcontrol_torque_v3.V3GovernorReason
V3LearnerRejectReason = latcontrol_torque_v3.V3LearnerRejectReason


def get_controller(car_name=TOYOTA.TOYOTA_RAV4):
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


def make_car_state(v_ego=20.0, steering_angle=0.0, steering_rate=0.0, steering_pressed=False):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.steeringAngleDeg = steering_angle
  CS.steeringRateDeg = steering_rate
  CS.steeringPressed = steering_pressed
  return CS


def update(controller, VM, CS, desired_curvature, *, active=True, steer_limited=False, curvature_limited=False):
  params = log.LiveParametersData.new_message()
  return controller.update(active, CS, VM, params, steer_limited, desired_curvature, make_pose(), curvature_limited, 0.2)


def test_v3_schema_fields_exist():
  torque_log = log.ControlsState.LateralTorqueState.new_message()
  adaptive_log = torque_log.init('adaptiveTorqueState')

  adaptive_log.rawTargetLateralAccel = 0.2
  adaptive_log.delayLeadLateralAccel = 0.3
  adaptive_log.feedbackCorrection = 0.01
  adaptive_log.trimCorrection = 0.02
  adaptive_log.learnerResponseScale = 1.05
  adaptive_log.governorReason = int(V3GovernorReason.SLEW_LIMITED)
  adaptive_log.actualLateralJerk = 0.4

  assert adaptive_log.delayLeadLateralAccel == pytest.approx(0.3)
  assert adaptive_log.learnerResponseScale == pytest.approx(1.05)
  assert adaptive_log.governorReason == V3GovernorReason.SLEW_LIMITED


def test_v3_controller_alias_matches_controller_symbol():
  assert LatControlTorqueV3 is LatControlTorque


def test_v3_uses_shared_torque_extension_hook():
  controller, VM = get_controller()
  CS = make_car_state(v_ego=20.0)
  base_factor = controller.torque_params.latAccelFactor

  class FakeExtension:
    def __init__(self):
      self.last_v_ego = 0.0
      self.updated = False

    def update_override_torque_params(self, torque_params):
      self.updated = True
      torque_params.latAccelFactor = base_factor * 1.1
      return True

  controller.extension = FakeExtension()

  update(controller, VM, CS, 0.001)

  assert controller.extension.updated
  assert controller.extension.last_v_ego == pytest.approx(CS.vEgo)
  assert controller.torque_params.latAccelFactor == pytest.approx(base_factor * 1.1)


def test_v3_delay_leads_curve_entry_and_release():
  controller, VM = get_controller()
  CS = make_car_state(v_ego=20.0)

  for _ in range(8):
    update(controller, VM, CS, 0.0)

  _, _, entry_log = update(controller, VM, CS, 0.001)
  entry_adaptive = entry_log.adaptiveTorqueState
  assert entry_adaptive.delayLeadLateralAccel > entry_adaptive.rawTargetLateralAccel
  assert abs(entry_log.output) > 0.0

  for _ in range(60):
    _, _, curve_log = update(controller, VM, CS, 0.001)

  _, _, exit_log = update(controller, VM, CS, 0.0)
  exit_adaptive = exit_log.adaptiveTorqueState
  assert exit_adaptive.delayLeadLateralAccel < exit_adaptive.rawTargetLateralAccel
  assert abs(exit_log.output) < abs(curve_log.output)


def test_v3_toyota_high_rate_governor_softens_rapid_steering():
  normal_controller, normal_vm = get_controller()
  high_rate_controller, high_rate_vm = get_controller()
  normal_cs = make_car_state(v_ego=20.0, steering_rate=0.0)
  high_rate_cs = make_car_state(v_ego=20.0, steering_rate=95.0)

  _, _, normal_log = update(normal_controller, normal_vm, normal_cs, 0.01)
  _, _, high_rate_log = update(high_rate_controller, high_rate_vm, high_rate_cs, 0.01)

  assert high_rate_log.adaptiveTorqueState.governorReason & V3GovernorReason.TOYOTA_HIGH_RATE
  assert abs(high_rate_log.output) < abs(normal_log.output)
  assert high_rate_log.adaptiveTorqueState.outputCap < normal_log.adaptiveTorqueState.outputCap


def test_v3_driver_override_releases_with_bounded_decay():
  controller, VM = get_controller()
  CS = make_car_state(v_ego=20.0)

  for _ in range(20):
    _, _, active_log = update(controller, VM, CS, 0.002)
  assert abs(active_log.output) > 0.0

  override_cs = make_car_state(v_ego=20.0, steering_pressed=True)
  _, _, override_log = update(controller, VM, override_cs, 0.002)

  assert override_log.adaptiveTorqueState.governorReason & V3GovernorReason.DRIVER_OVERRIDE
  assert abs(override_log.output) < abs(active_log.output)
  assert override_log.adaptiveTorqueState.sampleRejectReason & V3LearnerRejectReason.STEERING_PRESSED


def test_v3_signed_same_direction_limit_caps_output_and_reports_sample_rejection():
  controller, VM = get_controller()
  CS = make_car_state(v_ego=20.0)
  controller.set_steering_actuator_feedback(
    SteeringActuatorFeedback(True, True, SteeringLimitReason.ACTUATOR_MISMATCH, -0.8, -0.5, 0.3, True, False)
  )

  _, _, lac_log = update(controller, VM, CS, 0.01, steer_limited=True)
  adaptive = lac_log.adaptiveTorqueState

  assert adaptive.governorReason & V3GovernorReason.SAME_DIRECTION_LIMIT
  assert adaptive.sampleRejectReason & V3LearnerRejectReason.STEER_LIMITED
  assert adaptive.outputCap < 1.0
  assert not adaptive.learningFrozen


def test_v3_does_not_run_session_response_learning():
  controller, VM = get_controller()
  CS = make_car_state(v_ego=20.0)

  assert not hasattr(controller, "learner")

  for _ in range(180):
    _, _, lac_log = update(controller, VM, CS, 0.001)

  adaptive = lac_log.adaptiveTorqueState
  assert adaptive.sampleAccepted
  assert adaptive.modelConfidence == pytest.approx(0.0)
  assert adaptive.learnerResponseScale == pytest.approx(1.0)
  assert adaptive.authorityScale == pytest.approx(1.0)
  assert adaptive.trimCorrection == pytest.approx(0.0)


def test_v3_invalid_input_zeroes_output_and_logs_fault():
  controller, VM = get_controller()
  CS = make_car_state(v_ego=20.0)

  _, _, lac_log = update(controller, VM, CS, float("nan"))

  assert lac_log.output == 0.0
  assert lac_log.adaptiveTorqueState.governorReason & V3GovernorReason.INVALID
  assert lac_log.adaptiveTorqueState.sampleRejectReason & V3LearnerRejectReason.INACTIVE
