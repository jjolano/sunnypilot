import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.fallback import CustomStackFallbackWrapper
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V1, SUNNYPILOT_CURRENT


def make_output(a_target=0.0, source="cruise"):
  return LongitudinalStackOutput(
    a_target=a_target,
    should_stop=False,
    has_lead=False,
    source=source,
    allow_throttle=True,
    allow_brake=True,
    speeds=tuple(10.0 for _ in range(CONTROL_N)),
    accels=tuple(a_target for _ in range(CONTROL_N)),
    jerks=tuple(0.0 for _ in range(CONTROL_N)),
  )


def test_wrapper_actuates_custom_when_output_is_valid():
  wrapper = CustomStackFallbackWrapper()
  custom_output = make_output(0.2, "custom")
  fallback_output = make_output(0.0, "fallback")

  result = wrapper.update(True, lambda: custom_output, lambda: fallback_output, accel_limits=(-2.0, 2.0))

  assert result.output is custom_output
  assert result.actuated_stack == CUSTOM_V1
  assert result.shadow_stack == SUNNYPILOT_CURRENT
  assert result.shadow_output is fallback_output
  assert not result.fallback_latched
  assert not result.fallback_triggered
  assert wrapper.fallback_reason == ""


def test_wrapper_latches_fallback_on_invalid_custom_output_once():
  wrapper = CustomStackFallbackWrapper()
  fallback_output = make_output(0.0, "fallback")
  custom_calls = 0

  def invalid_custom():
    nonlocal custom_calls
    custom_calls += 1
    return make_output(3.0, "custom")

  first = wrapper.update(True, invalid_custom, lambda: fallback_output, accel_limits=(-2.0, 2.0))
  second = wrapper.update(True, invalid_custom, lambda: fallback_output, accel_limits=(-2.0, 2.0))

  assert first.output is fallback_output
  assert first.actuated_stack == SUNNYPILOT_CURRENT
  assert first.shadow_stack == CUSTOM_V1
  assert first.fallback_latched
  assert first.fallback_triggered
  assert first.fallback_reason == "a_target_above_limits"
  assert second.output is fallback_output
  assert second.fallback_latched
  assert not second.fallback_triggered
  assert custom_calls == 1


def test_wrapper_resets_latch_when_selfdrive_disabled():
  wrapper = CustomStackFallbackWrapper()
  fallback_output = make_output(0.0, "fallback")
  invalid_output = make_output(3.0, "custom")
  valid_output = make_output(0.1, "custom")

  tripped = wrapper.update(True, lambda: invalid_output, lambda: fallback_output, accel_limits=(-2.0, 2.0))
  disabled = wrapper.update(False, lambda: valid_output, lambda: fallback_output, accel_limits=(-2.0, 2.0))
  recovered = wrapper.update(True, lambda: valid_output, lambda: fallback_output, accel_limits=(-2.0, 2.0))

  assert tripped.fallback_latched
  assert not disabled.fallback_latched
  assert disabled.actuated_stack == SUNNYPILOT_CURRENT
  assert recovered.output is valid_output
  assert recovered.actuated_stack == CUSTOM_V1
  assert not recovered.fallback_latched


def test_wrapper_latches_on_custom_exception():
  wrapper = CustomStackFallbackWrapper()
  fallback_output = make_output(0.0, "fallback")

  def custom_raises():
    raise ValueError("boom")

  result = wrapper.update(True, custom_raises, lambda: fallback_output, accel_limits=(-2.0, 2.0))

  assert result.output is fallback_output
  assert result.fallback_latched
  assert result.fallback_triggered
  assert result.fallback_reason == "custom_exception"


def test_wrapper_raises_if_fallback_stack_is_invalid():
  wrapper = CustomStackFallbackWrapper()

  with pytest.raises(RuntimeError, match="fallback_stack_invalid:a_target_above_limits"):
    wrapper.update(True, lambda: make_output(0.0, "custom"), lambda: make_output(3.0, "fallback"), accel_limits=(-2.0, 2.0))
