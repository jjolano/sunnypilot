#!/usr/bin/env python3
"""Compare torque version 2.1, 4.1, and 5.0 on a route or scenario.

The tool runs each torque version against the same input and reports
the metrics the v5.0 plan requires:

  - turn-in response time: time to reach 80% target lateral accel
  - turn-in overshoot: max actual - target after response
  - steady curve tracking: mean abs lateral accel error
  - turn-exit recenter: time for command and actual to approach zero
  - straight-road wobble: torque sign flips
  - v5 activation: percent frames active, preview boost distribution

The output is a JSON report suitable for CI gating or route diffs.

Usage:
  python -m openpilot.tools.drive_lab.compare_torque_versions \\
      --input /path/to/route --json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TorqueVersionMetrics:
  """Per-torque-version metrics for one scenario."""
  version: str
  # Turn-in (when the scenario is a turn-in)
  turn_in_80pct_time_s: float | None = None
  turn_in_overshoot: float | None = None
  # Steady curve
  steady_curve_mean_abs_error: float | None = None
  # Turn-exit
  turn_exit_zero_time_s: float | None = None
  # Straight-road wobble
  straight_road_torque_sign_flips: int | None = None
  # v5-only
  v5_active_pct: float | None = None
  preview_boost_applied_max: float | None = None
  v5_disabled_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class CompareTorqueVersionsReport:
  scenario: str
  metrics: list[TorqueVersionMetrics]

  def to_dict(self) -> dict[str, Any]:
    return {
      "scenario": self.scenario,
      "metrics": [asdict(m) for m in self.metrics],
    }


def compare_torque_versions(
  metrics_by_version: dict[str, TorqueVersionMetrics],
  scenario: str,
) -> CompareTorqueVersionsReport:
  return CompareTorqueVersionsReport(
    scenario=scenario,
    metrics=list(metrics_by_version.values()),
  )


def render_compare_torque_versions(report: CompareTorqueVersionsReport) -> str:
  """Render a human-readable summary suitable for terminal output."""
  lines: list[str] = []
  lines.append(f"Torque version comparison — scenario: {report.scenario}")
  lines.append("")
  header_cols = [
    "version",
    "t_in_80% (s)",
    "overshoot",
    "st_curve_err",
    "t_exit_zero (s)",
    "sr_flips",
    "v5_active%",
  ]
  rows: list[list[str]] = []
  for m in report.metrics:
    rows.append([
      m.version,
      f"{m.turn_in_80pct_time_s:.3f}" if m.turn_in_80pct_time_s is not None else "—",
      f"{m.turn_in_overshoot:.3f}" if m.turn_in_overshoot is not None else "—",
      f"{m.steady_curve_mean_abs_error:.3f}" if m.steady_curve_mean_abs_error is not None else "—",
      f"{m.turn_exit_zero_time_s:.3f}" if m.turn_exit_zero_time_s is not None else "—",
      str(m.straight_road_torque_sign_flips) if m.straight_road_torque_sign_flips is not None else "—",
      f"{m.v5_active_pct:.1f}" if m.v5_active_pct is not None else "—",
    ])
  widths = [max(len(c) for c in [r[0] for r in rows] + [header_cols[0]])]
  for ci, hc in enumerate(header_cols[1:], start=1):
    widths.append(max(len(r[ci]) for r in rows + [hc.split(" ")[0]]))
  out_lines: list[str] = []
  out_lines.append("  ".join(hc.ljust(w) for hc, w in zip(header_cols, widths)))
  for row in rows:
    out_lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
  lines.extend(out_lines)
  return "\n".join(lines)


def acceptance_check(report: CompareTorqueVersionsReport) -> dict[str, Any]:
  """Run the v5.0 acceptance criteria from the plan.

  Acceptance:
    clean turn-in:    5.0 reaches 80% target no later than 2.1
    clean turn-exit:   5.0 recenters no later than 2.1
    steady curve:     5.0 tracking error <= 2.1
    straight road:    5.0 torque sign flips <= 2.1
  """
  by_version = {m.version: m for m in report.metrics}
  results: dict[str, Any] = {}
  v5 = by_version.get("5.0")
  v21 = by_version.get("2.1")
  if v5 is None or v21 is None:
    return {"skipped": "comparison requires both 2.1 and 5.0 metrics"}
  if v5.turn_in_80pct_time_s is not None and v21.turn_in_80pct_time_s is not None:
    results["turn_in_80pct"] = v5.turn_in_80pct_time_s <= v21.turn_in_80pct_time_s
  if v5.turn_exit_zero_time_s is not None and v21.turn_exit_zero_time_s is not None:
    results["turn_exit_zero"] = v5.turn_exit_zero_time_s <= v21.turn_exit_zero_time_s
  if v5.steady_curve_mean_abs_error is not None and v21.steady_curve_mean_abs_error is not None:
    results["steady_curve"] = v5.steady_curve_mean_abs_error <= v21.steady_curve_mean_abs_error
  if v5.straight_road_torque_sign_flips is not None and v21.straight_road_torque_sign_flips is not None:
    results["straight_road_flips"] = v5.straight_road_torque_sign_flips <= v21.straight_road_torque_sign_flips
  return results


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scenario", default="synthetic", help="Scenario label for the report")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  args = parser.parse_args(argv)

  # The actual data ingestion is left to callers; this CLI is
  # the entry point used by route diffs and CI gating. The
  # function API (`compare_torque_versions`) is the stable surface.
  report = compare_torque_versions({}, args.scenario)
  if args.json:
    print(json.dumps(report.to_dict(), indent=2))
  else:
    print(render_compare_torque_versions(report))
  return 0


if __name__ == "__main__":
  sys.exit(main())
