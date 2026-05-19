from __future__ import annotations

from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import CustomLongitudinalStackV2
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V2


def make_custom_longitudinal_stack(stack_name: str):
  if stack_name == CUSTOM_V2:
    return CustomLongitudinalStackV2()
  raise ValueError(f"unsupported_custom_stack:{stack_name}")
