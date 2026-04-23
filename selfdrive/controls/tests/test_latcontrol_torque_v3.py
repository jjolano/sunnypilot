import numpy as np
import sys
import types

from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.locationd.helpers import Measurement, Pose
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH

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

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v3 import LatControlTorque


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


def test_v3_logging_fields_are_populated():
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
  assert lac_log.version == 3
  adaptive_log = lac_log.adaptiveTorqueState
  assert adaptive_log.active
  assert adaptive_log.nominalOutput != 0.0
  assert abs(lac_log.output - (adaptive_log.nominalOutput + adaptive_log.assistOutput + adaptive_log.biasOutput)) < 1e-6


def test_v3_release_on_override():
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


def test_v3_softens_low_speed_same_sign_unwind():
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
  assert adaptive_log.nominalOutput < 0.95
  assert lac_log.output < 0.9
