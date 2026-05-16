#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from openpilot.tools.drive_lab.lateral_event_report import (
  build_lateral_event_report,
  render_lateral_event_report,
  save_lateral_event_report,
)


def main() -> None:
  parser = argparse.ArgumentParser(description="Profile lateral slow-wander, rebound, and fast-reversal events from route logs.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--output", help="Write report JSON to this path")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--max-events", type=int, default=15, help="Maximum ranked events to report")
  args = parser.parse_args()

  from openpilot.tools.lib.logreader import LogReader, ReadMode

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  msgs = list(LogReader(args.route, default_mode=read_mode, sort_by_time=True))
  report = build_lateral_event_report(msgs, source=args.route, already_sorted=True, max_events=args.max_events)
  if args.output:
    save_lateral_event_report(report, args.output)
  print(json.dumps(report.to_dict(), indent=2) if args.json else render_lateral_event_report(report))


if __name__ == "__main__":
  main()
