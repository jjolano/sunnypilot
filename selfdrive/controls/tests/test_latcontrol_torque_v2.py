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

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v2 import LatControlTorque


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


def test_learning_state_survives_reengagement():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 30
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  pose = make_pose()

  for _ in range(90):
    controller.update(True, CS, VM, params, False, 3e-4, pose, False, 0.2)

  assert controller.authority_envelope.buckets
  saved_floors = {key: bucket.authority_floor for key, bucket in controller.authority_envelope.buckets.items()}

  for _ in range(40):
    controller.update(False, CS, VM, params, False, 0.0, pose, False, 0.2)

  assert controller.authority_envelope.phase.name == "IDLE"

  controller.update(True, CS, VM, params, False, 3e-4, pose, False, 0.2)
  current_floors = {key: bucket.authority_floor for key, bucket in controller.authority_envelope.buckets.items()}
  assert current_floors == saved_floors

  fresh_controller, _ = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  assert not fresh_controller.authority_envelope.buckets


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
  assert lac_log.v2Phase == log.ControlsState.LateralTorqueState.V2Phase.hold
  assert lac_log.v2PhaseGain == 1.0
  assert lac_log.v2AuthorityFloor > 0.0
  assert lac_log.v2NominalOutput != 0.0
