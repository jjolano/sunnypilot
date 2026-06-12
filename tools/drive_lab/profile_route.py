#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.log_profile import build_longitudinal_profile, render_profile, save_profile
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report


def main() -> None:
  parser = argparse.ArgumentParser(description="Build a longitudinal fuzzing profile from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write profile JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  profile = build_longitudinal_profile(msgs, source=args.route, already_sorted=True)
  print(output_report(profile, json_output=args.json, renderer=render_profile, output_path=args.output, save=save_profile))


if __name__ == "__main__":
  main()
