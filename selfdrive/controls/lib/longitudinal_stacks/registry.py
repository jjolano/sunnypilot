"""Custom Stack construction seam.

This registry is intentionally shallow while only `custom-2.0` exists.  It is
kept to isolate Custom Stack construction from Stack Selection and to give a
single place for future stack implementations without expanding selector logic.
"""

from __future__ import annotations

from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_experimental import CustomExperimentalLongitudinalStack
from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import CustomLongitudinalStackV2
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_EXPERIMENTAL, CUSTOM_V2


def make_custom_longitudinal_stack(stack_name: str):
  if stack_name == CUSTOM_V2:
    return CustomLongitudinalStackV2()
  if stack_name == CUSTOM_EXPERIMENTAL:
    return CustomExperimentalLongitudinalStack()
  raise ValueError(f"unsupported_custom_stack:{stack_name}")
