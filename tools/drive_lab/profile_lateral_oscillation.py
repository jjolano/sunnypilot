#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from openpilot.tools.drive_lab.lateral_oscillation_profile import (
  build_lateral_oscillation_profile,
  render_lateral_profile,
  save_lateral_profile,
)


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

  from openpilot.tools.lib.logreader import LogReader, ReadMode

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  msgs = list(LogReader(args.route, default_mode=read_mode, sort_by_time=True))
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
  if args.output:
    save_lateral_profile(profile, args.output)
  print(json.dumps(profile.to_dict(), indent=2) if args.json else render_lateral_profile(profile))


if __name__ == "__main__":
  main()
