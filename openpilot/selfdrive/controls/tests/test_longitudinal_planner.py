#!/usr/bin/env python3
import numpy as np

from opendbc.car.car_helpers import interfaces
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.selfdrive.controls.lib.longitudinal_planner import limit_accel_in_turns, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V


def _get_vm():
  """Return a VehicleModel for a Toyota RAV4, matching the test_latcontrol.py fixture pattern."""
  CarInterface = interfaces[TOYOTA.TOYOTA_RAV4]
  CP = CarInterface.get_non_essential_params(TOYOTA.TOYOTA_RAV4)
  return VehicleModel(CP)


class TestLimitAccelInTurns:

  def test_straight_does_not_reduce_target(self):
    """With zero steering and zero roll, the upper clip stays at the total-accel envelope."""
    VM = _get_vm()
    v_ego = 30.0
    a_target = [-5.0, 2.0]
    result = limit_accel_in_turns(v_ego, 0.0, a_target, VM, 0.0)
    a_total_max = float(np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V))
    assert result[0] == a_target[0]
    assert result[1] == min(a_target[1], a_total_max)

  def test_reduces_accel_in_high_curvature_turn(self):
    """At higher speed with nonzero steering, returned upper accel is lower than input."""
    VM = _get_vm()
    v_ego = 30.0
    a_target = [-5.0, 3.0]
    result = limit_accel_in_turns(v_ego, 100.0, a_target, VM, 0.0)
    assert result[1] < a_target[1]

  def test_uses_live_steer_ratio(self):
    """Changing VM steer ratio changes computed allowed acceleration in the expected direction.

    Higher steer ratio -> less curvature -> less lateral accel -> more longitudinal accel allowed.
    """
    VM = _get_vm()
    v_ego = 20.0
    a_target = [-5.0, 3.0]
    angle = 5.0

    VM.update_params(1.0, 13.0)
    result_low_ratio = limit_accel_in_turns(v_ego, angle, a_target, VM, 0.0)

    VM.update_params(1.0, 20.0)
    result_high_ratio = limit_accel_in_turns(v_ego, angle, a_target, VM, 0.0)

    assert result_high_ratio[1] > result_low_ratio[1]

  def test_no_nan_when_lateral_accel_exceeds_total_limit(self):
    """When abs(a_y) > a_total_max, upper accel becomes 0.0 rather than NaN."""
    VM = _get_vm()
    v_ego = 30.0
    a_target = [-5.0, 3.0]
    # Large steering angle to force a_y well beyond a_total_max
    result = limit_accel_in_turns(v_ego, 1000.0, a_target, VM, 0.0)
    assert np.isfinite(result[1])
    assert result[1] == 0.0
