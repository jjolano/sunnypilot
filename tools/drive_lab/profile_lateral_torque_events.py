#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.lateral_torque_event_report import (
  build_lateral_low_speed_report,
  build_lateral_torque_event_report,
  render_lateral_low_speed_report,
  render_lateral_torque_event_report,
  save_lateral_torque_event_report,
)
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile torque telemetry around lateral fast-reversal events from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--max-events", type=int, default=12, help="Maximum ranked torque events to report")
  parser.add_argument("--low-speed", action="store_true", help="Profile low-speed lateral tier metrics instead of fast torque events")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  if args.low_speed:
    report = build_lateral_low_speed_report(msgs, source=args.route, already_sorted=True)
    print(output_report(
      report,
      json_output=args.json,
      renderer=render_lateral_low_speed_report,
      output_path=args.output,
    ))
    return

  report = build_lateral_torque_event_report(msgs, source=args.route, already_sorted=True, max_events=args.max_events)
  print(output_report(
    report,
    json_output=args.json,
    renderer=render_lateral_torque_event_report,
    output_path=args.output,
    save=save_lateral_torque_event_report,
  ))


if __name__ == "__main__":
  main()
