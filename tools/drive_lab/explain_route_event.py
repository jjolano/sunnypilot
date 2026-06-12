#!/usr/bin/env python3
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.timeline import render_summary, select_event_time, summarize_window
from openpilot.tools.drive_lab.route_io import load_route_msgs


def main() -> None:
  parser = argparse.ArgumentParser(description="Explain planner/controller state around a route event.")
  parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--time", type=float, help="Event time in seconds from the first loaded log message")
  parser.add_argument("--nearest-bookmark", action="store_true", help="Use the nearest userBookmark event instead of the exact --time")
  parser.add_argument("--before", type=float, default=30.0, help="Seconds before the event to include")
  parser.add_argument("--after", type=float, default=30.0, help="Seconds after the event to include")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  args = parser.parse_args()

  msgs = load_route_msgs(args.route, qlog=args.qlog)
  event_time_s = select_event_time(msgs, args.time, args.nearest_bookmark)
  summary = summarize_window(msgs, event_time_s, args.before, args.after)
  print(render_summary(summary))


if __name__ == "__main__":
  main()
