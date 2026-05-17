from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import (
  LongitudinalStackOutput,
  validate_stack_output,
)
from openpilot.selfdrive.controls.lib.longitudinal_stacks.selector import CUSTOM_V1, SUNNYPILOT_CURRENT


StackUpdateFn = Callable[[], LongitudinalStackOutput]


@dataclass(frozen=True)
class FallbackUpdateResult:
  output: LongitudinalStackOutput
  actuated_stack: str
  shadow_stack: str
  shadow_output: LongitudinalStackOutput | None
  fallback_latched: bool
  fallback_reason: str
  fallback_triggered: bool


class CustomStackFallbackWrapper:
  def __init__(self, custom_stack: str = CUSTOM_V1, fallback_stack: str = SUNNYPILOT_CURRENT) -> None:
    self.custom_stack = custom_stack
    self.fallback_stack = fallback_stack
    self.fallback_latched = False
    self.fallback_reason = ""

  def reset(self) -> None:
    self.fallback_latched = False
    self.fallback_reason = ""

  def update(self, enabled: bool, custom_update: StackUpdateFn, fallback_update: StackUpdateFn,
             accel_limits: tuple[float | None, float | None] = (None, None),
             expected_trajectory_len: int = CONTROL_N) -> FallbackUpdateResult:
    fallback_output = self._validated_fallback_output(fallback_update, accel_limits, expected_trajectory_len)
    if not enabled:
      self.reset()
      return FallbackUpdateResult(
        output=fallback_output,
        actuated_stack=self.fallback_stack,
        shadow_stack="",
        shadow_output=None,
        fallback_latched=False,
        fallback_reason="",
        fallback_triggered=False,
      )

    if self.fallback_latched:
      return self._fallback_result(fallback_output, None, triggered=False)

    try:
      custom_output = custom_update()
    except Exception:
      self._latch("custom_exception")
      return self._fallback_result(fallback_output, None, triggered=True)

    validation = validate_stack_output(custom_output, accel_limits, expected_trajectory_len)
    if not validation.valid:
      self._latch(validation.reason)
      return self._fallback_result(fallback_output, custom_output if isinstance(custom_output, LongitudinalStackOutput) else None, triggered=True)

    return FallbackUpdateResult(
      output=custom_output,
      actuated_stack=self.custom_stack,
      shadow_stack=self.fallback_stack,
      shadow_output=fallback_output,
      fallback_latched=False,
      fallback_reason="",
      fallback_triggered=False,
    )

  def _latch(self, reason: str) -> None:
    self.fallback_latched = True
    self.fallback_reason = reason

  def _fallback_result(self, fallback_output: LongitudinalStackOutput,
                       shadow_output: LongitudinalStackOutput | None,
                       triggered: bool) -> FallbackUpdateResult:
    return FallbackUpdateResult(
      output=fallback_output,
      actuated_stack=self.fallback_stack,
      shadow_stack=self.custom_stack if shadow_output is not None else "",
      shadow_output=shadow_output,
      fallback_latched=True,
      fallback_reason=self.fallback_reason,
      fallback_triggered=triggered,
    )

  def _validated_fallback_output(self, fallback_update: StackUpdateFn,
                                 accel_limits: tuple[float | None, float | None],
                                 expected_trajectory_len: int) -> LongitudinalStackOutput:
    try:
      fallback_output = fallback_update()
    except Exception as exc:
      raise RuntimeError("fallback_stack_exception") from exc

    validation = validate_stack_output(fallback_output, accel_limits, expected_trajectory_len)
    if not validation.valid:
      raise RuntimeError(f"fallback_stack_invalid:{validation.reason}")
    return fallback_output
