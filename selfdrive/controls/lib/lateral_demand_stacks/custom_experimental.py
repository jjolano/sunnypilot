from openpilot.selfdrive.controls.lib.lateral_demand_stacks.interface import (
  LateralDemandStackInputs,
  LateralDemandStackOutput,
  validate_lateral_demand_stack_output,
)
from openpilot.selfdrive.controls.lib.lateral_demand_stacks.custom_v2 import CustomV2LateralDemandStack


class CustomExperimentalLateralDemandStack(CustomV2LateralDemandStack):
  NAME = "custom-experimental"
  VERSION = "experimental"

  def __init__(self, dt: float) -> None:
    super().__init__(dt)
    self._stage: str = "v2_baseline"

  @property
  def stage(self) -> str:
    return self._stage

  def update(self, inputs: LateralDemandStackInputs) -> LateralDemandStackOutput:
    output = super().update(inputs)
    self._stage = "v2_baseline"
    staged = LateralDemandStackOutput(
      requested_stack=self.NAME,
      resolved_stack=self.NAME,
      fallback_reason="",
      version=self.VERSION,
      legacy=output.legacy,
      profile=output.profile,
      debug={**output.debug, "experimental_stage": self._stage},
    )
    validation = validate_lateral_demand_stack_output(staged)
    if not validation.valid:
      raise RuntimeError(f"custom-experimental stack produced invalid output: {validation.reason}")
    return staged
