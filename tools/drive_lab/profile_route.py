#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from openpilot.tools.drive_lab.log_profile import build_longitudinal_profile, render_profile, save_profile


def main() -> None:
  parser = argparse.ArgumentParser(description="Build a longitudinal fuzzing profile from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write profile JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  args = parser.parse_args()

  from openpilot.tools.lib.logreader import LogReader, ReadMode

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  msgs = list(LogReader(args.route, default_mode=read_mode, sort_by_time=True))
  profile = build_longitudinal_profile(msgs, source=args.route)
  if args.output:
    save_profile(profile, args.output)
  print(json.dumps(profile.to_dict(), indent=2) if args.json else render_profile(profile))


if __name__ == "__main__":
  main()
