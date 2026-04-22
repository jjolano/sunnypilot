from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.mock.generators import generate_livePose
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.locationd.helpers import Pose
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v2 import LatControlTorque


def get_controller(car_name):
  CarInterface = interfaces[car_name]
  CP = CarInterface.get_non_essential_params(car_name)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
  CI = CarInterface(CP, CP_SP)
  sunnypilot_interfaces.setup_interfaces(CI)
  CP_SP = convert_to_capnp(CP_SP)
  VM = VehicleModel(CP)
  controller = LatControlTorque(CP.as_reader(), CP_SP.as_reader(), CI, DT_CTRL)
  return controller, VM


def test_learning_state_survives_reengagement():
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)

  CS = car.CarState.new_message()
  CS.vEgo = 30
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  lp = generate_livePose()
  pose = Pose.from_live_pose(lp.livePose)

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
