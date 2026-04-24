import pytest
from cereal import car, custom
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import (
  LAUNCH_BREAKAWAY_A_EGO,
  LAUNCH_BREAKAWAY_ACCEL,
  LAUNCH_BREAKAWAY_MAX_TIME,
  LAUNCH_BREAKAWAY_MIN_TIME,
  LAUNCH_BREAKAWAY_V_EGO,
  LAUNCH_ENVELOPE_MAX_ACCEL,
  LAUNCH_ENVELOPE_MIN_ACCEL,
  LAUNCH_SHOULD_STOP_HOLD_TIME,
  LongControl,
  LongCtrlState,
  apply_launch_envelope,
  get_launch_envelope_blend,
  launch_breakaway_active,
  launch_should_stop_hold_active,
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


def test_launch_breakaway_holds_until_response_or_timeout():
  assert launch_breakaway_active(0.0, 0.0, 0.0)
  assert launch_breakaway_active(LAUNCH_BREAKAWAY_V_EGO - 1e-3, 0.0, LAUNCH_BREAKAWAY_MIN_TIME)
  assert not launch_breakaway_active(LAUNCH_BREAKAWAY_V_EGO, 0.0, 0.0)
  assert not launch_breakaway_active(0.0, LAUNCH_BREAKAWAY_A_EGO, LAUNCH_BREAKAWAY_MIN_TIME)
  assert not launch_breakaway_active(0.0, 0.0, LAUNCH_BREAKAWAY_MAX_TIME)


def test_launch_should_stop_hold_only_applies_immediately_after_release():
  assert launch_should_stop_hold_active(0.0, False, 0.0)
  assert not launch_should_stop_hold_active(LAUNCH_BREAKAWAY_V_EGO, False, 0.0)
  assert not launch_should_stop_hold_active(0.0, True, 0.0)
  assert not launch_should_stop_hold_active(0.0, False, LAUNCH_SHOULD_STOP_HOLD_TIME)


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
  assert not loc.launch_breakaway_done
  assert output_accel == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)


def test_starting_state_launch_arms_and_caps_after_stop():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=0.2, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.starting
  assert loc.launch_envelope_active
  assert not loc.launch_breakaway_done
  assert output_accel == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)


def test_breakaway_holds_until_response_then_hands_off_to_taper():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  accel_limits = (-3.0, 2.0)

  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=accel_limits)
  assert output_accel == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)

  for _ in range(int(LAUNCH_BREAKAWAY_MIN_TIME / DT_CTRL) + 1):
    output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=accel_limits)
  assert output_accel == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)

  taper_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.1), a_target=1.0, should_stop=False, accel_limits=accel_limits)

  assert loc.launch_breakaway_done
  assert taper_accel == pytest.approx(LAUNCH_ENVELOPE_MAX_ACCEL)
  assert taper_accel > LAUNCH_ENVELOPE_MIN_ACCEL


def test_breakaway_times_out_when_response_never_arrives():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  accel_limits = (-3.0, 2.0)

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=accel_limits)
  for _ in range(int(LAUNCH_BREAKAWAY_MAX_TIME / DT_CTRL) + 1):
    output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=accel_limits)

  assert loc.launch_breakaway_done
  assert LAUNCH_ENVELOPE_MAX_ACCEL <= output_accel < LAUNCH_BREAKAWAY_ACCEL


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


def test_should_stop_reassertion_is_ignored_during_launch_hold():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=0.05, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.pid
  assert loc.launch_envelope_active
  assert output_accel == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)


def test_should_stop_reassertion_returns_after_launch_hold():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  for _ in range(int(LAUNCH_SHOULD_STOP_HOLD_TIME / DT_CTRL) + 1):
    loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))

  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=0.0, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0
