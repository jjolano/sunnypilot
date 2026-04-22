import pytest
from cereal import car, custom
from openpilot.selfdrive.controls.lib.longcontrol import (
  LAUNCH_ENVELOPE_MAX_ACCEL,
  LAUNCH_ENVELOPE_MIN_ACCEL,
  LongControl,
  LongCtrlState,
  apply_launch_envelope,
  get_launch_envelope_blend,
  long_control_state_trans,
)


class TestLongControlStateTransition:
  def test_stay_stopped(self):
    CP = car.CarParams.new_message()
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=1.0, should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=1.0, should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off


def test_engage():
  CP = car.CarParams.new_message()
  CP_SP = custom.CarParamsSP.new_message()
  active = True
  current_state = LongCtrlState.off
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=True, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=True, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=True)
  assert next_state == LongCtrlState.stopping
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid


def test_starting():
  CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
  CP_SP = custom.CarParamsSP.new_message()
  active = True
  current_state = LongCtrlState.starting
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=0.1, should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.starting
  next_state = long_control_state_trans(CP, CP_SP, active, current_state, v_ego=1.0, should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid


def make_car_params(**kwargs):
  CP = car.CarParams.new_message()
  CP.stopAccel = -2.0
  CP.stoppingDecelRate = 0.8
  CP.vEgoStarting = 0.25
  CP.startAccel = 1.0
  CP.longitudinalTuning.kpBP = [0.0]
  CP.longitudinalTuning.kpV = [1.0]
  CP.longitudinalTuning.kiBP = [0.0]
  CP.longitudinalTuning.kiV = [0.0]
  for key, value in kwargs.items():
    setattr(CP, key, value)
  return CP


def make_car_state(v_ego=0.0, a_ego=0.0, brake_pressed=False, cruise_standstill=False):
  CS = car.CarState.new_message()
  CS.vEgo = v_ego
  CS.aEgo = a_ego
  CS.brakePressed = brake_pressed
  CS.cruiseState.standstill = cruise_standstill
  return CS


def test_launch_envelope_blend_fades_by_time_and_speed():
  assert get_launch_envelope_blend(0.0, 0.0) == pytest.approx(1.0)
  assert 0.0 < get_launch_envelope_blend(0.3, 0.25) < 1.0
  assert get_launch_envelope_blend(0.7, 0.0) == pytest.approx(0.0)
  assert get_launch_envelope_blend(0.0, 0.6) == pytest.approx(0.0)


def test_apply_launch_envelope_only_shapes_positive_accel():
  accel_limits = (-3.0, 2.0)
  assert apply_launch_envelope(-0.2, accel_limits, 0.0, 0.0) == pytest.approx(-0.2)
  assert apply_launch_envelope(0.0, accel_limits, 0.0, 0.0) == pytest.approx(0.0)
  assert apply_launch_envelope(1.0, accel_limits, 0.0, 0.0) == pytest.approx(LAUNCH_ENVELOPE_MAX_ACCEL)


def test_pid_launch_arms_and_caps_after_stop():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.pid
  assert loc.launch_envelope_active
  assert output_accel >= LAUNCH_ENVELOPE_MIN_ACCEL
  assert output_accel == pytest.approx(LAUNCH_ENVELOPE_MAX_ACCEL)


def test_starting_state_launch_arms_and_caps_after_stop():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=0.2, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.starting
  assert loc.launch_envelope_active
  assert output_accel >= LAUNCH_ENVELOPE_MIN_ACCEL
  assert output_accel == pytest.approx(LAUNCH_ENVELOPE_MAX_ACCEL)


def test_launch_envelope_cancels_when_stop_returns():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  assert loc.launch_envelope_active

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=0.0, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0
