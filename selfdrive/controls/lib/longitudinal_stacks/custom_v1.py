from __future__ import annotations

from dataclasses import replace

from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V1


class CustomLongitudinalStackV1:
  stack_name = CUSTOM_V1

  def update(self, sunnypilot_output: LongitudinalStackOutput) -> LongitudinalStackOutput:
    # Initial shell: preserve current behavior while giving custom-v1 a real module boundary.
    debug = dict(sunnypilot_output.debug)
    debug["custom_stack"] = self.stack_name
    debug["custom_v1_mode"] = "passthrough"
    return replace(sunnypilot_output, debug=debug)
