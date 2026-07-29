#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.lateral_performance_gate import (
  build_lateral_performance_gate,
  build_lateral_performance_gate_ab_report,
  render_lateral_performance_gate,
  render_lateral_performance_gate_ab_report,
  save_lateral_performance_gate,
)
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report


def main() -> None:
  parser = argparse.ArgumentParser(description="Run the combined lateral performance gate on route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("candidate", nargs="?", help="Optional candidate route/log for baseline-vs-candidate comparison")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--strict-lane-state", action="store_true", help="Do not use qlog-safe handling for unknown lane-change state")
  parser.add_argument("--baseline-label", default="baseline", help="Label for the baseline route when comparing two routes")
  parser.add_argument("--candidate-label", default="candidate", help="Label for the candidate route when comparing two routes")
  args = parser.parse_args()

  qlog_safe_lane_policy = not args.strict_lane_state
  baseline_msgs = load_route_msgs(args.route, qlog=args.qlog)
  if args.candidate:
    candidate_msgs = load_route_msgs(args.candidate, qlog=args.qlog)
    report = build_lateral_performance_gate_ab_report(
      baseline_msgs,
      candidate_msgs,
      baseline_source=args.baseline_label,
      candidate_source=args.candidate_label,
      already_sorted=True,
      qlog_safe_lane_policy=qlog_safe_lane_policy,
    )
    renderer = render_lateral_performance_gate_ab_report
  else:
    report = build_lateral_performance_gate(
      baseline_msgs,
      source=args.route,
      already_sorted=True,
      qlog_safe_lane_policy=qlog_safe_lane_policy,
    )
    renderer = render_lateral_performance_gate

  print(output_report(
    report,
    json_output=args.json,
    renderer=renderer,
    output_path=args.output,
    save=save_lateral_performance_gate,
  ))


if __name__ == "__main__":
  main()
