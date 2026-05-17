import math

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput, validate_stack_output


def make_output(**kwargs):
  values = {
    "a_target": 0.0,
    "should_stop": False,
    "has_lead": False,
    "source": "cruise",
    "allow_throttle": True,
    "allow_brake": True,
    "speeds": tuple(10.0 for _ in range(CONTROL_N)),
    "accels": tuple(0.0 for _ in range(CONTROL_N)),
    "jerks": tuple(0.0 for _ in range(CONTROL_N)),
    "fcw": False,
    "debug": {},
  }
  values.update(kwargs)
  return LongitudinalStackOutput(**values)


def assert_invalid(output, reason, accel_limits=(-2.0, 2.0)):
  validation = validate_stack_output(output, accel_limits=accel_limits)
  assert not validation.valid
  assert validation.reason == reason


def test_valid_stack_output_passes_contract():
  validation = validate_stack_output(make_output(a_target=0.5), accel_limits=(-2.0, 2.0))

  assert validation.valid
  assert validation.reason == ""


def test_invalid_output_type_fails_contract():
  assert_invalid(object(), "invalid_output_type")


def test_non_bool_control_flags_fail_contract():
  assert_invalid(make_output(should_stop=1), "invalid_should_stop")
  assert_invalid(make_output(allow_throttle=1), "invalid_allow_throttle")
  assert_invalid(make_output(fcw=0), "invalid_fcw")


def test_missing_source_and_invalid_debug_fail_contract():
  assert_invalid(make_output(source=None), "missing_source")
  assert_invalid(make_output(debug=[]), "invalid_debug")


def test_non_finite_a_target_fails_contract():
  assert_invalid(make_output(a_target=math.nan), "non_finite_a_target")


def test_a_target_outside_limits_fails_contract():
  assert_invalid(make_output(a_target=-3.0), "a_target_below_limits")
  assert_invalid(make_output(a_target=3.0), "a_target_above_limits")


def test_invalid_accel_limits_fail_contract():
  assert_invalid(make_output(), "invalid_accel_limits", accel_limits=(-2.0, math.inf))
  assert_invalid(make_output(), "inverted_accel_limits", accel_limits=(2.0, -2.0))


def test_invalid_trajectories_fail_contract():
  assert_invalid(make_output(speeds=(1.0,)), "invalid_speeds_length")
  assert_invalid(make_output(accels=tuple(math.nan for _ in range(CONTROL_N))), "non_finite_accels")
  assert_invalid(make_output(jerks="bad"), "invalid_jerks")
