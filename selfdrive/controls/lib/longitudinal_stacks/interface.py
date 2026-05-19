from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N


@dataclass(frozen=True)
class LongitudinalStackOutput:
  a_target: float
  should_stop: bool
  has_lead: bool
  source: object
  allow_throttle: bool
  allow_brake: bool
  speeds: tuple[float, ...]
  accels: tuple[float, ...]
  jerks: tuple[float, ...]
  fcw: bool = False
  debug: Mapping[str, Any] = field(default_factory=dict)
  seed_intent: str = ""
  seed_reason: str = ""


@dataclass(frozen=True)
class StackOutputValidation:
  valid: bool
  reason: str = ""


def validate_stack_output(output: object, accel_limits: tuple[float | None, float | None] = (None, None),
                          expected_trajectory_len: int = CONTROL_N) -> StackOutputValidation:
  if not isinstance(output, LongitudinalStackOutput):
    return StackOutputValidation(False, "invalid_output_type")

  bool_fields = ("should_stop", "has_lead", "allow_throttle", "allow_brake", "fcw")
  for field_name in bool_fields:
    if not isinstance(getattr(output, field_name), bool):
      return StackOutputValidation(False, f"invalid_{field_name}")

  if output.source is None:
    return StackOutputValidation(False, "missing_source")

  if not isinstance(output.debug, Mapping):
    return StackOutputValidation(False, "invalid_debug")
  if not isinstance(output.seed_intent, str):
    return StackOutputValidation(False, "invalid_seed_intent")
  if not isinstance(output.seed_reason, str):
    return StackOutputValidation(False, "invalid_seed_reason")

  if not _finite_number(output.a_target):
    return StackOutputValidation(False, "non_finite_a_target")

  limits_reason = _accel_limits_invalid_reason(accel_limits)
  if limits_reason:
    return StackOutputValidation(False, limits_reason)

  lower, upper = accel_limits
  if lower is not None and output.a_target < float(lower):
    return StackOutputValidation(False, "a_target_below_limits")
  if upper is not None and output.a_target > float(upper):
    return StackOutputValidation(False, "a_target_above_limits")

  for field_name in ("speeds", "accels", "jerks"):
    reason = _trajectory_invalid_reason(field_name, getattr(output, field_name), expected_trajectory_len)
    if reason:
      return StackOutputValidation(False, reason)

  return StackOutputValidation(True)


def _finite_number(value: object) -> bool:
  return isinstance(value, Real) and math.isfinite(float(value))


def _accel_limits_invalid_reason(accel_limits: object) -> str:
  if not isinstance(accel_limits, tuple) or len(accel_limits) != 2:
    return "invalid_accel_limits"
  lower, upper = accel_limits
  for value in (lower, upper):
    if value is not None and not _finite_number(value):
      return "invalid_accel_limits"
  if lower is not None and upper is not None and float(lower) > float(upper):
    return "inverted_accel_limits"
  return ""


def _trajectory_invalid_reason(field_name: str, trajectory: object, expected_len: int) -> str:
  if not isinstance(trajectory, Sequence) or isinstance(trajectory, (bytes, str)):
    return f"invalid_{field_name}"
  if len(trajectory) != expected_len:
    return f"invalid_{field_name}_length"
  if not all(_finite_number(value) for value in trajectory):
    return f"non_finite_{field_name}"
  return ""
