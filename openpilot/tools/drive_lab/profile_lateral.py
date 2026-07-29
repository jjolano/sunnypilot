#!/usr/bin/env python3
"""Build a lateral fuzzing profile from route logs.

Usage:
  uv run python tools/drive_lab/profile_lateral.py ROUTE [--output profile.json] [--json]

Extracts lateral-relevant data (speed, curvature, lane-line confidence, roll)
from route logs and produces a ``LateralProfile`` with percentile ranges that
can be used for profile-guided lateral fuzzing.
"""
from __future__ import annotations

import argparse

from openpilot.tools.drive_lab.log_profile import (
    build_lateral_profile,
    render_lateral_profile,
    save_lateral_profile,
)
from openpilot.tools.drive_lab.route_io import load_route_msgs, output_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lateral fuzzing profile from route logs.")
    parser.add_argument("route", help="Route, segment range, log file, or URL accepted by LogReader")
    parser.add_argument("--output", help="Write profile JSON to this path")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
    parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
    args = parser.parse_args()

    msgs = load_route_msgs(args.route, qlog=args.qlog)
    profile = build_lateral_profile(msgs, source=args.route, already_sorted=True)
    print(output_report(profile, json_output=args.json, renderer=render_lateral_profile,
                         output_path=args.output, save=save_lateral_profile))


if __name__ == "__main__":
    main()
