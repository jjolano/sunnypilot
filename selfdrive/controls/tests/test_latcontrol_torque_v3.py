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
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_authority import AuthorityBand
from openpilot.sunnypilot.selfdrive.controls.lib.torque_v3_model import TorqueModelMode

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

from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v3 import LatControlTorque, LatControlTorqueV3


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


def test_v3_controller_alias_matches_controller_symbol():
  assert LatControlTorqueV3 is LatControlTorque


def test_v3_native_torque_controller_logs_model_state():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 20.0
  params = log.LiveParametersData.new_message()

  _, _, lac_log = controller.update(True, CS, VM, params, False, 5e-4, make_pose(), False, 0.2)

  assert lac_log.version == 3
  assert lac_log.adaptiveTorqueState.modelMode in (TorqueModelMode.native, TorqueModelMode.learned)
  assert lac_log.adaptiveTorqueState.authorityScale > 0.0


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
