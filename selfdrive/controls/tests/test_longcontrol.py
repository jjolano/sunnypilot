import pytest
from cereal import car, custom
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import (
  LAUNCH_BREAKAWAY_A_EGO,
  LAUNCH_BREAKAWAY_ACCEL,
  LAUNCH_BREAKAWAY_BASE_ACCEL,
  LAUNCH_BREAKAWAY_MAX_TIME,
  LAUNCH_BREAKAWAY_MIN_TIME,
  LAUNCH_BREAKAWAY_V_EGO,
  LAUNCH_ENVELOPE_MAX_ACCEL,
  LAUNCH_ENVELOPE_MIN_ACCEL,
  LAUNCH_SHOULD_STOP_HOLD_TIME,
  LongControl,
  LongCtrlState,
  apply_launch_envelope,
  get_launch_breakaway_accel,
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


def test_stopping_softens_stop_accel_while_still_rolling():
  CP = make_car_params(stopAccel=-2.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -0.799

  output_accel = loc.update(True, make_car_state(v_ego=0.2), a_target=-0.2, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert output_accel == pytest.approx(-0.8)


def test_stopping_preserves_gentler_platform_stop_accel_while_rolling():
  CP = make_car_params(stopAccel=-0.55)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -0.549

  output_accel = loc.update(True, make_car_state(v_ego=0.2), a_target=-0.2, should_stop=True, accel_limits=(-3.0, 2.0))

  assert output_accel == pytest.approx(-0.55)


def test_stopping_allows_full_stop_accel_after_standstill_confirmed():
  CP = make_car_params(stopAccel=-2.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -0.799

  output_accel = loc.update(True, make_car_state(v_ego=0.2, cruise_standstill=True), a_target=-0.2, should_stop=True, accel_limits=(-3.0, 2.0))

  assert output_accel == pytest.approx(-0.799 - CP.stoppingDecelRate * DT_CTRL)
  assert output_accel < -0.8


def test_stopping_allows_full_stop_accel_at_near_zero_speed():
  CP = make_car_params(stopAccel=-2.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -0.799

  output_accel = loc.update(True, make_car_state(v_ego=0.03), a_target=-0.2, should_stop=True, accel_limits=(-3.0, 2.0))

  assert output_accel == pytest.approx(-0.799 - CP.stoppingDecelRate * DT_CTRL)
  assert output_accel < -0.8


def test_stopping_recovers_to_soft_stop_accel_if_standstill_signal_drops():
  CP = make_car_params(stopAccel=-2.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  loc.last_output_accel = -1.0

  output_accel = loc.update(True, make_car_state(v_ego=0.2), a_target=-0.2, should_stop=True, accel_limits=(-3.0, 2.0))

  assert output_accel == pytest.approx(-0.8)


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
  assert launch_should_stop_hold_active(0.0, 0.0, False, 0.0, LAUNCH_ENVELOPE_MIN_ACCEL + 1e-3)
  assert not launch_should_stop_hold_active(LAUNCH_BREAKAWAY_V_EGO, 0.0, False, 0.0, LAUNCH_ENVELOPE_MIN_ACCEL)
  assert not launch_should_stop_hold_active(0.0, LAUNCH_BREAKAWAY_A_EGO, False, 0.0, LAUNCH_ENVELOPE_MIN_ACCEL)
  assert not launch_should_stop_hold_active(0.0, 0.0, True, 0.0, LAUNCH_ENVELOPE_MIN_ACCEL)
  assert not launch_should_stop_hold_active(0.0, 0.0, False, LAUNCH_SHOULD_STOP_HOLD_TIME, LAUNCH_ENVELOPE_MIN_ACCEL)
  assert not launch_should_stop_hold_active(0.0, 0.0, False, 0.0, LAUNCH_ENVELOPE_MIN_ACCEL)


def test_launch_breakaway_accel_scales_with_target_accel():
  accel_limits = (-3.0, 2.0)

  assert get_launch_breakaway_accel(0.0, accel_limits) == pytest.approx(0.0)
  assert get_launch_breakaway_accel(-0.2, accel_limits) == pytest.approx(0.0)
  assert get_launch_breakaway_accel(LAUNCH_ENVELOPE_MIN_ACCEL - 1e-3, accel_limits) == pytest.approx(0.0)
  mild_launch = get_launch_breakaway_accel(0.18, accel_limits)
  assert LAUNCH_ENVELOPE_MIN_ACCEL < mild_launch < LAUNCH_BREAKAWAY_BASE_ACCEL
  assert get_launch_breakaway_accel(0.35, accel_limits) == pytest.approx(LAUNCH_BREAKAWAY_BASE_ACCEL)
  assert get_launch_breakaway_accel(0.50, accel_limits) == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)
  assert get_launch_breakaway_accel(1.0, accel_limits) == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)
  assert get_launch_breakaway_accel(1.0, (-3.0, 0.25)) == pytest.approx(0.25)


def test_clear_runway_launch_allows_stronger_breakaway_and_faster_ramp_out():
  accel_limits = (-3.0, 2.0)

  assert LAUNCH_BREAKAWAY_ACCEL >= 0.55
  assert LAUNCH_ENVELOPE_MAX_ACCEL >= 0.55
  assert get_launch_breakaway_accel(1.0, accel_limits) == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)
  assert apply_launch_envelope(1.0, accel_limits, 0.0, 0.0) == pytest.approx(LAUNCH_ENVELOPE_MAX_ACCEL)
  assert get_launch_envelope_blend(0.0, 0.4) == pytest.approx(0.0)


def test_adaptive_breakaway_base_for_mild_launch():
  accel_limits = (-3.0, 2.0)
  # Mild/neutral launch (a_target around 0.35) should get base breakaway 0.62-0.65
  breakaway = get_launch_breakaway_accel(0.35, accel_limits)
  assert 0.62 <= breakaway <= 0.65


def test_adaptive_breakaway_upper_cap_for_clear_runway():
  accel_limits = (-3.0, 2.0)
  # Clear-runway launch with clearly positive target should get upper cap ~0.70
  assert get_launch_breakaway_accel(0.50, accel_limits) == pytest.approx(0.70, abs=0.01)
  assert get_launch_breakaway_accel(1.0, accel_limits) == pytest.approx(0.70, abs=0.01)


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

  a_target = 1.0
  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=a_target, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.pid
  assert loc.launch_envelope_active
  assert not loc.launch_breakaway_done
  assert output_accel == pytest.approx(get_launch_breakaway_accel(a_target, (-3.0, 2.0)))


def test_lead_launch_bypasses_no_lead_breakaway_envelope():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  a_target = 1.0
  output_accel = loc.update(
    True, make_car_state(v_ego=0.0, a_ego=a_target), a_target=a_target, should_stop=False, accel_limits=(-3.0, 2.0), has_lead=True,
  )

  assert loc.long_control_state == LongCtrlState.pid
  assert not loc.launch_envelope_active
  assert output_accel == pytest.approx(a_target)


@pytest.mark.parametrize("a_target", [0.0, LAUNCH_ENVELOPE_MIN_ACCEL - 1e-3])
def test_pid_launch_uses_minimum_breakaway_for_non_negative_target(a_target):
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=a_target, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.pid
  assert loc.launch_envelope_active
  assert not loc.launch_breakaway_done
  assert output_accel == pytest.approx(get_launch_breakaway_accel(LAUNCH_ENVELOPE_MIN_ACCEL, (-3.0, 2.0)))


def test_pid_launch_does_not_arm_for_negative_target():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=-0.2, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.pid
  assert not loc.launch_envelope_active
  assert output_accel < 0.0


def test_active_launch_cancels_when_target_drops_non_positive():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  assert loc.launch_envelope_active

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=-0.2, should_stop=False, accel_limits=(-3.0, 2.0))

  assert not loc.launch_envelope_active
  assert output_accel < 0.0


def test_starting_state_launch_arms_and_caps_after_stop():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  a_target = 0.2
  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=a_target, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.starting
  assert loc.launch_envelope_active
  assert not loc.launch_breakaway_done
  assert output_accel == pytest.approx(get_launch_breakaway_accel(a_target, (-3.0, 2.0)))


def test_starting_state_lead_launch_uses_planner_target():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  a_target = 0.2
  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=a_target, should_stop=False, accel_limits=(-3.0, 2.0), has_lead=True)

  assert loc.long_control_state == LongCtrlState.starting
  assert not loc.launch_envelope_active
  assert output_accel == pytest.approx(a_target)


def test_starting_state_launch_uses_minimum_breakaway_for_neutral_target():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=0.0, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.starting
  assert loc.launch_envelope_active
  assert not loc.launch_breakaway_done
  assert output_accel == pytest.approx(get_launch_breakaway_accel(LAUNCH_ENVELOPE_MIN_ACCEL, (-3.0, 2.0)))


def test_starting_state_launch_does_not_command_start_accel_for_negative_target():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=-0.2, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.starting
  assert not loc.launch_envelope_active
  assert output_accel < 0.0


def test_starting_state_below_launch_threshold_uses_minimum_breakaway():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  a_target = LAUNCH_ENVELOPE_MIN_ACCEL - 1e-3
  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=a_target, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.starting
  assert loc.launch_envelope_active
  assert not loc.launch_breakaway_done
  assert output_accel == pytest.approx(get_launch_breakaway_accel(LAUNCH_ENVELOPE_MIN_ACCEL, (-3.0, 2.0)))


def test_starting_state_active_launch_cancels_when_target_drops_non_positive():
  CP = make_car_params(startingState=True, startAccel=1.0)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0), a_target=0.2, should_stop=False, accel_limits=(-3.0, 2.0))
  assert loc.long_control_state == LongCtrlState.starting
  assert loc.launch_envelope_active

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=-0.2, should_stop=False, accel_limits=(-3.0, 2.0))

  assert not loc.launch_envelope_active
  assert output_accel < 0.0


def test_breakaway_holds_until_response_then_hands_off_to_taper():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  accel_limits = (-3.0, 2.0)

  a_target = 1.0
  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=a_target, should_stop=False, accel_limits=accel_limits)
  assert output_accel == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)

  for _ in range(int(LAUNCH_BREAKAWAY_MIN_TIME / DT_CTRL) + 1):
    output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=a_target, should_stop=False, accel_limits=accel_limits)
  assert output_accel == pytest.approx(LAUNCH_BREAKAWAY_ACCEL)

  taper_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.1), a_target=a_target, should_stop=False, accel_limits=accel_limits)

  assert loc.launch_breakaway_done
  assert taper_accel == pytest.approx(LAUNCH_ENVELOPE_MAX_ACCEL)
  assert taper_accel > LAUNCH_ENVELOPE_MIN_ACCEL


@pytest.mark.parametrize("a_target", [0.0, 0.05])
def test_breakaway_handoff_preserves_minimum_accel_for_neutral_target(a_target):
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  accel_limits = (-3.0, 2.0)

  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=a_target, should_stop=False, accel_limits=accel_limits)
  assert output_accel == pytest.approx(LAUNCH_ENVELOPE_MIN_ACCEL)

  for _ in range(int(LAUNCH_BREAKAWAY_MIN_TIME / DT_CTRL) + 1):
    output_accel = loc.update(True, make_car_state(v_ego=0.05, a_ego=LAUNCH_ENVELOPE_MIN_ACCEL), a_target=a_target,
                              should_stop=False, accel_limits=accel_limits)

  assert loc.launch_breakaway_done
  handoff_accel = output_accel
  assert handoff_accel > 0.0

  for _ in range(8):
    output_accel = loc.update(True, make_car_state(v_ego=0.05, a_ego=LAUNCH_ENVELOPE_MIN_ACCEL), a_target=a_target,
                              should_stop=False, accel_limits=accel_limits)

  assert handoff_accel > output_accel > 0.0


def test_taper_update_reuses_launch_blend_for_cap_and_reset(monkeypatch):
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping
  accel_limits = (-3.0, 2.0)
  a_target = 1.0

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=a_target, should_stop=False, accel_limits=accel_limits)
  for _ in range(int(LAUNCH_BREAKAWAY_MIN_TIME / DT_CTRL) + 1):
    loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=a_target, should_stop=False, accel_limits=accel_limits)

  calls = 0

  def counted_blend(v_ego, launch_elapsed):
    nonlocal calls
    calls += 1
    return get_launch_envelope_blend(v_ego, launch_elapsed)

  monkeypatch.setattr("openpilot.selfdrive.controls.lib.longcontrol.get_launch_envelope_blend", counted_blend)

  taper_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.1), a_target=a_target, should_stop=False, accel_limits=accel_limits)

  assert calls == 1
  assert taper_accel == pytest.approx(LAUNCH_ENVELOPE_MAX_ACCEL)


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
  assert LAUNCH_ENVELOPE_MAX_ACCEL <= output_accel <= 1.0


def test_launch_envelope_cancels_when_negative_stop_returns():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  assert loc.launch_envelope_active

  output_accel = loc.update(True, make_car_state(v_ego=0.0), a_target=-0.2, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0


def test_positive_should_stop_reassertion_is_ignored_during_launch_hold():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=LAUNCH_ENVELOPE_MIN_ACCEL + 1e-3, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.pid
  assert loc.launch_envelope_active
  assert output_accel == pytest.approx(get_launch_breakaway_accel(LAUNCH_ENVELOPE_MIN_ACCEL + 1e-3, (-3.0, 2.0)))


def test_neutral_should_stop_reassertion_cancels_launch_hold():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=0.0, should_stop=False, accel_limits=(-3.0, 2.0))
  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=0.0, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0


def test_minimum_accel_should_stop_reassertion_cancels_launch_hold():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=LAUNCH_ENVELOPE_MIN_ACCEL, should_stop=False, accel_limits=(-3.0, 2.0))
  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=LAUNCH_ENVELOPE_MIN_ACCEL, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0


def test_neutral_should_stop_reassertion_returns_while_breakaway_waits_for_response():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  for _ in range(int((LAUNCH_BREAKAWAY_MIN_TIME + 0.1) / DT_CTRL)):
    loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))

  assert loc.launch_breakaway_elapsed < LAUNCH_BREAKAWAY_MAX_TIME
  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=0.0, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0


def test_should_stop_reassertion_returns_after_launch_response():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  for _ in range(int(LAUNCH_BREAKAWAY_MIN_TIME / DT_CTRL) + 1):
    loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))

  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=LAUNCH_BREAKAWAY_A_EGO), a_target=0.0, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0


def test_should_stop_reassertion_returns_after_breakaway_timeout():
  CP = make_car_params(startingState=False)
  CP_SP = custom.CarParamsSP.new_message()
  loc = LongControl(CP, CP_SP)
  loc.long_control_state = LongCtrlState.stopping

  loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))
  for _ in range(int(LAUNCH_BREAKAWAY_MAX_TIME / DT_CTRL) + 1):
    loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=1.0, should_stop=False, accel_limits=(-3.0, 2.0))

  output_accel = loc.update(True, make_car_state(v_ego=0.0, a_ego=0.0), a_target=0.0, should_stop=True, accel_limits=(-3.0, 2.0))

  assert loc.long_control_state == LongCtrlState.stopping
  assert not loc.launch_envelope_active
  assert output_accel <= 0.0
