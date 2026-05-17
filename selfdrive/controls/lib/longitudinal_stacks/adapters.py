from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.longitudinal_stacks.interface import LongitudinalStackOutput


def planner_state_to_stack_output(planner: object, has_lead: bool, source: object | None = None,
                                  debug: dict[str, Any] | None = None) -> LongitudinalStackOutput:
  mpc = getattr(planner, "mpc", None)
  if source is None:
    source = getattr(mpc, "source", getattr(planner, "source", ""))
  return LongitudinalStackOutput(
    a_target=float(getattr(planner, "output_a_target", 0.0)),
    should_stop=bool(getattr(planner, "output_should_stop", False)),
    has_lead=bool(has_lead),
    source=source,
    allow_throttle=bool(getattr(planner, "allow_throttle", True)),
    allow_brake=True,
    speeds=_trajectory_tuple(getattr(planner, "v_desired_trajectory", ()), CONTROL_N),
    accels=_trajectory_tuple(getattr(planner, "a_desired_trajectory", ()), CONTROL_N),
    jerks=_trajectory_tuple(getattr(planner, "j_desired_trajectory", ()), CONTROL_N),
    fcw=bool(getattr(planner, "fcw", False)),
    debug=debug or {},
  )


def apply_stack_output_to_planner(planner: object, output: LongitudinalStackOutput) -> None:
  import numpy as np

  planner.output_a_target = float(output.a_target)
  planner.output_should_stop = bool(output.should_stop)
  planner.allow_throttle = bool(output.allow_throttle)
  planner.fcw = bool(output.fcw)
  planner.v_desired_trajectory = np.asarray(output.speeds, dtype=float)
  planner.a_desired_trajectory = np.asarray(output.accels, dtype=float)
  planner.j_desired_trajectory = np.asarray(output.jerks, dtype=float)


def _trajectory_tuple(values: object, expected_len: int) -> tuple[float, ...]:
  if not isinstance(values, Sequence):
    return tuple(0.0 for _ in range(expected_len))
  return tuple(float(value) for value in values)
