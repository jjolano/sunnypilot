from __future__ import annotations

from dataclasses import dataclass

import numpy as np


COL_TIME = 0
COL_SPEED = 3
COL_LEAD_SPEED = 4
COL_ACCEL = 5
COL_D_REL = 6

LEAD_PRESENT_D_REL = 199.0
LAUNCH_LEAD_MOVING_SPEED = 0.2
LAUNCH_EGO_MOVING_SPEED = 0.2
LAUNCH_WINDOW_SPEED = 5.0
LAUNCH_WINDOW_TIME = 8.0
LEAD_APPROACH_RESPONSE_GRACE = 2.0
LEAD_APPROACH_CLOSE_TIME_GAP = 2.5


@dataclass(frozen=True)
class ExpectedRange:
  low: float
  high: float
  unit: str

  def contains(self, value: float) -> bool:
    return self.low <= value <= self.high

  def render(self) -> str:
    return f"{_format_value(self.low)} to {_format_value(self.high)} {self.unit}".strip()


@dataclass(frozen=True)
class MetricComparison:
  area: str
  metric: str
  label: str
  current: float
  expected: ExpectedRange
  passed: bool


@dataclass(frozen=True)
class ScenarioComparison:
  title: str
  kind: str
  valid: bool
  comparisons: list[MetricComparison]

  @property
  def passed(self) -> bool:
    return self.valid and all(comparison.passed for comparison in self.comparisons)


@dataclass(frozen=True)
class BehaviorOutline:
  area: str
  current_behavior: str
  expected_behavior: str


EXPECTED_RANGES = {
  "launch_delay": ExpectedRange(0.0, 1.2, "s"),
  "launch_mean_accel": ExpectedRange(0.20, 1.25, "m/s^2"),
  "launch_peak_accel": ExpectedRange(0.25, 1.80, "m/s^2"),
  "max_abs_jerk": ExpectedRange(0.0, 8.0, "m/s^3"),
  "max_closing_speed": ExpectedRange(0.0, 4.5, "m/s"),
  "min_time_gap": ExpectedRange(1.2, 12.0, "s"),
  "min_lead_gap": ExpectedRange(4.0, 120.0, "m"),
  "stop_mean_decel": ExpectedRange(-1.50, -0.20, "m/s^2"),
  "stop_peak_decel": ExpectedRange(-2.80, -0.20, "m/s^2"),
  "final_lead_gap": ExpectedRange(4.0, 12.0, "m"),
}

BEHAVIOR_OUTLINE = (
  BehaviorOutline(
    "Launch from stop",
    "Fixed launch envelope can feel delayed or abrupt depending on target accel and vehicle response.",
    "Low-latency breakaway followed by a smooth taper into the planner target.",
  ),
  BehaviorOutline(
    "Lead pullaway",
    "Stopped-lead creep and pullaway depend on gap excess, lead confidence, and pullaway prediction.",
    "Hold while the lead is stationary, then release gently once the lead clearly opens the gap.",
  ),
  BehaviorOutline(
    "Lead approach",
    "MPC and follow-gap logic can be safe while still feeling late or uneven to the driver.",
    "Earlier, smoother convergence to the target gap with stable closing time.",
  ),
  BehaviorOutline(
    "Stopping behind lead",
    "Creep-to-stop and stopped-lead gap-fill can vary with model probability, lead speed, and gap excess.",
    "Predictable decel taper that lands near the expected final gap without last-second brake pulses.",
  ),
  BehaviorOutline(
    "No-lead model stop",
    "E2E stop comfort shaping may still allow late braking when the model endpoint shortens.",
    "Earlier light decel when needed, without unnecessary braking when runway is adequate.",
  ),
)

LAUNCH_KINDS = {
  "lead_pullaway",
  "udacity_acc_green_light_launch",
  "openpilot_resume_from_stop",
  "udacity_acc_approach_from_stop",
  "route_replay_none",
}
STOP_KINDS = {
  "stopped_lead_approach",
  "udacity_acc_stopped_lead",
  "udacity_acc_lead_decel_to_stop",
  "udacity_acc_lead_decel_to_stop_2ms2",
  "udacity_acc_stop_and_go",
  "udacity_acc_stop_and_go_10mph",
  "openpilot_stopped_lead_25ms_120m",
  "openpilot_stopped_lead_20ms_90m",
  "openpilot_lead_decel_1ms2",
  "openpilot_lead_decel_2ms2",
  "openpilot_lead_decel_3ms2",
  "openpilot_lead_decel_3plus_ms2",
  "commonroad_zam_acc_1_1",
  "ncap_ccrs_70",
  "ncap_ccrs_90",
  "ncap_ccrs_110",
  "ncap_ccrs_130",
}
LEAD_APPROACH_KINDS = {
  "slower_cut_in",
  "lead_occlusion",
  "udacity_acc_slower_lead",
  "udacity_acc_oscillating_lead",
  "udacity_acc_accel_while_lead_decel_mild",
  "udacity_acc_accel_while_lead_decel_hard",
  "openpilot_slower_cut_in",
  "commonroad_zam_acc_1_2",
  "commonroad_zam_acc_1_3",
}


def compare_scenario_output(
  kind: str,
  output: np.ndarray,
  *,
  commanded_accel: np.ndarray | None = None,
  jerk_window: int = 1,
) -> list[MetricComparison]:
  output = _validated_output(output)
  jerk_accel = output[:, COL_ACCEL]
  if commanded_accel is not None:
    commanded_accel = np.asarray(commanded_accel, dtype=float)
    if commanded_accel.shape == jerk_accel.shape:
      jerk_accel = commanded_accel
  comparisons: list[MetricComparison] = []
  if kind in LAUNCH_KINDS:
    comparisons.extend(_launch_comparisons(output, jerk_accel, jerk_window))
  if kind in LEAD_APPROACH_KINDS:
    comparisons.extend(_lead_approach_comparisons(output))
  if kind in STOP_KINDS:
    comparisons.extend(_stop_comparisons(output, jerk_accel, jerk_window))
  return comparisons


def render_behavior_outline() -> str:
  lines = [
    "| Area | Current Behavior | Expected Behavior |",
    "|---|---|---|",
  ]
  lines.extend(
    f"| {outline.area} | {outline.current_behavior} | {outline.expected_behavior} |"
    for outline in BEHAVIOR_OUTLINE
  )
  return "\n".join(lines)


def render_comparison_table(comparisons: list[MetricComparison]) -> str:
  lines = [
    "| Area | Metric | Current | Expected | Result |",
    "|---|---:|---:|---:|---|",
  ]
  if not comparisons:
    lines.append("| none | none | n/a | n/a | pass |")
    return "\n".join(lines)

  for comparison in comparisons:
    current = f"{_format_value(comparison.current)} {comparison.expected.unit}".strip()
    result = "pass" if comparison.passed else "fail"
    lines.append(f"| {comparison.area} | {comparison.label} | {current} | {comparison.expected.render()} | {result} |")
  return "\n".join(lines)


def _launch_comparisons(output: np.ndarray, jerk_accel: np.ndarray, jerk_window: int) -> list[MetricComparison]:
  t = output[:, COL_TIME]
  speed = output[:, COL_SPEED]
  lead_speed = output[:, COL_LEAD_SPEED]
  accel = output[:, COL_ACCEL]
  lead_move_time = _first_time(t, lead_speed > LAUNCH_LEAD_MOVING_SPEED, default=t[0])
  ego_move_time = _first_time(t, (t >= lead_move_time) & (speed > LAUNCH_EGO_MOVING_SPEED), default=t[-1])
  launch_window = (t >= ego_move_time) & (t <= lead_move_time + LAUNCH_WINDOW_TIME) & (speed <= LAUNCH_WINDOW_SPEED)
  launch_accels = accel[launch_window]
  positive_launch_accels = launch_accels[launch_accels > 0.0]
  if positive_launch_accels.size:
    launch_accels = positive_launch_accels
  if launch_accels.size == 0:
    launch_accels = accel

  return [
    _comparison("Launch", "launch_delay", "launch delay", ego_move_time - lead_move_time),
    _comparison("Launch", "launch_mean_accel", "launch mean accel", float(np.mean(launch_accels))),
    _comparison("Launch", "launch_peak_accel", "launch peak accel", float(np.max(launch_accels))),
    _comparison("Launch", "max_abs_jerk", "max jerk", _max_abs_jerk(t, jerk_accel, jerk_window)),
  ]


def _lead_approach_comparisons(output: np.ndarray) -> list[MetricComparison]:
  t = output[:, COL_TIME]
  speed = output[:, COL_SPEED]
  lead_speed = output[:, COL_LEAD_SPEED]
  d_rel = output[:, COL_D_REL]
  lead_present = (d_rel < LEAD_PRESENT_D_REL) & (speed > 1.0)
  if not np.any(lead_present):
    return []

  lead_start_time = _first_time(t, lead_present, default=t[0])
  response_window = lead_present & (t >= lead_start_time + LEAD_APPROACH_RESPONSE_GRACE)
  if not np.any(response_window):
    response_window = lead_present

  closing_window = response_window & (d_rel / np.maximum(speed, 0.1) <= LEAD_APPROACH_CLOSE_TIME_GAP)
  if not np.any(closing_window):
    closing_window = response_window

  closing_speed = np.maximum(speed[closing_window] - lead_speed[closing_window], 0.0)
  time_gap = d_rel[response_window] / np.maximum(speed[response_window], 0.1)
  return [
    _comparison("Lead approach", "max_closing_speed", "max closing speed", float(np.max(closing_speed))),
    _comparison("Lead approach", "min_time_gap", "min time gap", float(np.min(time_gap))),
    _comparison("Lead approach", "min_lead_gap", "min lead gap", float(np.min(d_rel[response_window]))),
  ]


def _stop_comparisons(output: np.ndarray, jerk_accel: np.ndarray, jerk_window: int) -> list[MetricComparison]:
  speed = output[:, COL_SPEED]
  accel = output[:, COL_ACCEL]
  d_rel = output[:, COL_D_REL]
  decel_samples = accel[(speed > 0.1) & (accel < 0.0)]
  if decel_samples.size == 0:
    decel_samples = np.array([0.0])

  stopped_or_slow = (speed <= 1.0) & (d_rel < LEAD_PRESENT_D_REL)
  final_gap = float(d_rel[stopped_or_slow][-1]) if np.any(stopped_or_slow) else float(d_rel[-1])
  return [
    _comparison("Stopping", "stop_mean_decel", "stop mean accel", float(np.mean(decel_samples))),
    _comparison("Stopping", "stop_peak_decel", "stop peak decel", float(np.min(decel_samples))),
    _comparison("Stopping", "final_lead_gap", "final lead gap", final_gap),
    _comparison("Stopping", "max_abs_jerk", "max jerk", _max_abs_jerk(output[:, COL_TIME], jerk_accel, jerk_window)),
  ]


def _comparison(area: str, metric: str, label: str, current: float) -> MetricComparison:
  expected = EXPECTED_RANGES[metric]
  passed = expected.contains(current)
  return MetricComparison(area, metric, label, _rounded(current), expected, passed)


def _validated_output(output: np.ndarray) -> np.ndarray:
  output = np.asarray(output, dtype=float)
  if output.ndim != 2 or output.shape[1] <= COL_D_REL:
    raise ValueError(f"expected maneuver output with at least {COL_D_REL + 1} columns")
  if output.shape[0] == 0:
    raise ValueError("expected maneuver output with at least one row")
  return output


def _first_time(t: np.ndarray, mask: np.ndarray, default: float) -> float:
  matches = np.flatnonzero(mask)
  return float(t[int(matches[0])]) if matches.size else float(default)


def _max_abs_jerk(t: np.ndarray, accel: np.ndarray, window: int = 1) -> float:
  window = max(1, int(window))
  if accel.size <= window:
    return 0.0
  dt = t[window:] - t[:-window]
  accel_delta = accel[window:] - accel[:-window]
  valid = (dt > 1e-6) & np.isfinite(dt) & np.isfinite(accel_delta)
  if not np.any(valid):
    return 0.0
  return float(np.max(np.abs(accel_delta[valid] / dt[valid])))


def _rounded(value: float) -> float:
  return round(float(value), 3)


def _format_value(value: float) -> str:
  if abs(value) >= 10.0:
    return f"{value:.1f}"
  return f"{value:.3f}".rstrip("0").rstrip(".")
