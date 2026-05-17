from __future__ import annotations

from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v1 import CustomLongitudinalStackV1
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V1


def make_custom_longitudinal_stack(stack_name: str):
  if stack_name == CUSTOM_V1:
    return CustomLongitudinalStackV1()
  raise ValueError(f"unsupported_custom_stack:{stack_name}")
