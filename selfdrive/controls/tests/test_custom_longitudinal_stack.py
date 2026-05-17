import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v1 import CustomLongitudinalStackV1
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.registry import make_custom_longitudinal_stack
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V1


def make_output(debug=None):
  return LongitudinalStackOutput(
    a_target=-0.2,
    should_stop=False,
    has_lead=True,
    source="cruise",
    allow_throttle=True,
    allow_brake=True,
    speeds=tuple(10.0 for _ in range(CONTROL_N)),
    accels=tuple(-0.2 for _ in range(CONTROL_N)),
    jerks=tuple(0.0 for _ in range(CONTROL_N)),
    fcw=False,
    debug={} if debug is None else debug,
  )


def test_custom_v1_preserves_sunnypilot_output_and_marks_debug_boundary():
  original_debug = {"adapter": "sunnypilot-current"}
  sunnypilot_output = make_output(debug=original_debug)

  custom_output = CustomLongitudinalStackV1().update(sunnypilot_output)

  assert custom_output.a_target == sunnypilot_output.a_target
  assert custom_output.should_stop == sunnypilot_output.should_stop
  assert custom_output.has_lead == sunnypilot_output.has_lead
  assert custom_output.source == sunnypilot_output.source
  assert custom_output.allow_throttle == sunnypilot_output.allow_throttle
  assert custom_output.allow_brake == sunnypilot_output.allow_brake
  assert custom_output.speeds == sunnypilot_output.speeds
  assert custom_output.accels == sunnypilot_output.accels
  assert custom_output.jerks == sunnypilot_output.jerks
  assert custom_output.fcw == sunnypilot_output.fcw
  assert custom_output.debug == {
    "adapter": "sunnypilot-current",
    "custom_stack": CUSTOM_V1,
    "custom_v1_mode": "passthrough",
  }
  assert original_debug == {"adapter": "sunnypilot-current"}


def test_custom_stack_factory_builds_custom_v1():
  stack = make_custom_longitudinal_stack(CUSTOM_V1)

  assert isinstance(stack, CustomLongitudinalStackV1)
  assert stack.stack_name == CUSTOM_V1


def test_custom_stack_factory_rejects_unknown_stack():
  with pytest.raises(ValueError, match="unsupported_custom_stack:custom-9.9"):
    make_custom_longitudinal_stack("custom-9.9")
