from __future__ import annotations

from openpilot.selfdrive.controls.lib.longitudinal_stacks.custom_v2 import CustomLongitudinalStackV2
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_EXPERIMENTAL


class CustomExperimentalLongitudinalStack(CustomLongitudinalStackV2):
  NAME = CUSTOM_EXPERIMENTAL
  VERSION = "experimental"

  def __init__(self) -> None:
    super().__init__()
    self._stage: str = "v2_baseline"

  @property
  def stage(self) -> str:
    return self._stage

  def update(self, sunnypilot_output: LongitudinalStackOutput, scene=None, *,
             sm=None, v_ego=0.0, a_ego=0.0, gas_pressed=False, brake_pressed=False,
             steering_pressed=False, mpc_mode=0, personality=0,
             longitudinal_plan_source=0, accel_limits=(None, None)) -> LongitudinalStackOutput:
    output = super().update(
      sunnypilot_output, scene, sm=sm, v_ego=v_ego, a_ego=a_ego,
      gas_pressed=gas_pressed, brake_pressed=brake_pressed,
      steering_pressed=steering_pressed, mpc_mode=mpc_mode, personality=personality,
      longitudinal_plan_source=longitudinal_plan_source, accel_limits=accel_limits,
    )
    self._stage = "v2_baseline"
    return LongitudinalStackOutput(
      a_target=output.a_target,
      should_stop=output.should_stop,
      has_lead=output.has_lead,
      source=output.source,
      allow_throttle=output.allow_throttle,
      allow_brake=output.allow_brake,
      speeds=output.speeds,
      accels=output.accels,
      jerks=output.jerks,
      debug={**output.debug, "experimental_stage": self._stage},
    )

  def reset(self) -> None:
    super().reset()
    self._stage = "v2_baseline"
