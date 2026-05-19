import pytest

from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import CustomLongitudinalStackV2
from openpilot.selfdrive.controls.lib.longitudinal_stacks.registry import make_custom_longitudinal_stack
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2


def test_custom_stack_factory_builds_custom_v2():
  stack = make_custom_longitudinal_stack(CUSTOM_V2)

  assert isinstance(stack, CustomLongitudinalStackV2)
  assert stack.stack_name == CUSTOM_V2


def test_custom_stack_factory_rejects_removed_v1_stack():
  with pytest.raises(ValueError, match="unsupported_custom_stack:custom-1.0"):
    make_custom_longitudinal_stack("custom-1.0")


def test_custom_stack_factory_rejects_unknown_stack():
  with pytest.raises(ValueError, match="unsupported_custom_stack:custom-9.9"):
    make_custom_longitudinal_stack("custom-9.9")
