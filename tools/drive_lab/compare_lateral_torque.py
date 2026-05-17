#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from openpilot.tools.drive_lab.lateral_torque_event_report import (
  build_lateral_torque_ab_report,
  render_lateral_torque_ab_report,
)


def main() -> None:
  parser = argparse.ArgumentParser(description="Compare lateral torque tracking lag between two route logs.")
  parser.add_argument("baseline", help="Baseline route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("candidate", help="Candidate route, segment range, log file, or URL accepted by LogReader")
  parser.add_argument("--baseline-label", default="baseline", help="Label for the baseline route")
  parser.add_argument("--candidate-label", default="candidate", help="Label for the candidate route")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  args = parser.parse_args()

  from openpilot.tools.lib.logreader import LogReader, ReadMode

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  baseline_msgs = list(LogReader(args.baseline, default_mode=read_mode, sort_by_time=True))
  candidate_msgs = list(LogReader(args.candidate, default_mode=read_mode, sort_by_time=True))
  report = build_lateral_torque_ab_report(
    baseline_msgs,
    candidate_msgs,
    baseline_source=args.baseline_label,
    candidate_source=args.candidate_label,
    already_sorted=True,
  )
  print(json.dumps(report.to_dict(), indent=2) if args.json else render_lateral_torque_ab_report(report))


if __name__ == "__main__":
  main()
