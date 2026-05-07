import math

import numpy as np
import pytest

from cereal import car, log
from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.locationd.helpers import Measurement, Pose
from openpilot.sunnypilot.selfdrive.controls.lib import latcontrol_torque_v0
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH


LatControlTorque = latcontrol_torque_v0.LatControlTorque


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


@pytest.mark.parametrize("lat_delay", [0.0, -0.1, 1e-320, float("nan"), float("inf")])
def test_v0_invalid_lat_delay_uses_safe_delay(lat_delay):
  controller, VM = get_controller(TOYOTA.TOYOTA_COROLLA_TSS2)
  CS = car.CarState.new_message()
  CS.vEgo = 15.0
  CS.steeringPressed = False
  params = log.LiveParametersData.new_message()

  try:
    output_torque, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, make_pose(), False, lat_delay)
  except Exception as exc:
    pytest.fail(f"controller update raised {exc!r}")

  assert math.isfinite(output_torque)
  assert math.isfinite(lac_log.desiredLateralAccel)
  assert math.isfinite(lac_log.desiredLateralJerk)
