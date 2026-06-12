#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.lateral_oscillation_profile import (
  build_lateral_oscillation_profile,
  render_lateral_profile,
  save_lateral_profile,
)
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile slow lateral oscillation from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write profile JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--min-speed", type=float, default=8.0, help="Minimum speed for straight-road candidates in m/s")
  parser.add_argument("--max-raw-curvature", type=float, default=0.002, help="Maximum absolute raw model curvature for candidates")
  parser.add_argument("--window", type=float, default=30.0, help="Window length in seconds")
  parser.add_argument("--step", type=float, default=5.0, help="Window step in seconds")
  parser.add_argument("--max-windows", type=int, default=8, help="Maximum ranked windows to report")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  profile = build_lateral_oscillation_profile(
    msgs,
    source=args.route,
    already_sorted=True,
    min_speed=args.min_speed,
    max_raw_curvature=args.max_raw_curvature,
    window_s=args.window,
    step_s=args.step,
    max_windows=args.max_windows,
  )
  print(output_report(
    profile,
    json_output=args.json,
    renderer=render_lateral_profile,
    output_path=args.output,
    save=save_lateral_profile,
  ))


if __name__ == "__main__":
  main()
