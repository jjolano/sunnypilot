#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.lateral_disturbance_profile import (
  build_lateral_disturbance_profile,
  render_lateral_disturbance_profile,
  save_lateral_disturbance_profile,
)
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile lateral disturbance classification on route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  report = build_lateral_disturbance_profile(msgs, source=args.route, already_sorted=True)
  print(output_report(
    report,
    json_output=args.json,
    renderer=render_lateral_disturbance_profile,
    output_path=args.output,
    save=save_lateral_disturbance_profile,
  ))


if __name__ == "__main__":
  main()
