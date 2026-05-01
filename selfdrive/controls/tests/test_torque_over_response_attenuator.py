import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.torque_over_response_attenuator import (
  OVER_RESPONSE_ATTENUATION_MIN_SCALE,
  attenuate_same_direction_over_response,
)


def test_positive_over_response_trims_inward_torque():
  nominal_torque = 0.8

  result = attenuate_same_direction_over_response(nominal_torque, desired_lateral_accel=0.4, actual_lateral_accel=1.1)

  assert result == pytest.approx(nominal_torque * OVER_RESPONSE_ATTENUATION_MIN_SCALE)


def test_negative_over_response_trims_inward_torque():
  nominal_torque = -0.8

  result = attenuate_same_direction_over_response(nominal_torque, desired_lateral_accel=-0.4, actual_lateral_accel=-1.1)

  assert result == pytest.approx(nominal_torque * OVER_RESPONSE_ATTENUATION_MIN_SCALE)


def test_over_response_margin_keeps_nominal_torque():
  nominal_torque = 0.8

  result = attenuate_same_direction_over_response(nominal_torque, desired_lateral_accel=0.4, actual_lateral_accel=0.52)

  assert result == nominal_torque


def test_under_response_keeps_nominal_torque():
  nominal_torque = 0.8

  result = attenuate_same_direction_over_response(nominal_torque, desired_lateral_accel=0.8, actual_lateral_accel=0.4)

  assert result == nominal_torque


def test_corrective_torque_keeps_nominal_torque():
  nominal_torque = -0.8

  result = attenuate_same_direction_over_response(nominal_torque, desired_lateral_accel=0.4, actual_lateral_accel=1.1)

  assert result == nominal_torque


def test_attenuation_does_not_flip_torque_sign():
  nominal_torque = 0.05

  result = attenuate_same_direction_over_response(nominal_torque, desired_lateral_accel=0.4, actual_lateral_accel=2.0)

  assert 0.0 < result < nominal_torque
