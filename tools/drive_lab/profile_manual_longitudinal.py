#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import isfinite
from typing import Any

from openpilot.tools.drive_lab.manual_longitudinal_profile import (
  ManualSample,
  build_route_profile,
  render_manual_style_summary,
  summarize_manual_style,
)
from openpilot.tools.drive_lab.timeline import msg_payload, msg_time_s, msg_type, safe_get
from openpilot.tools.lib.logreader import LogReader, ReadMode


def main() -> None:
  parser = argparse.ArgumentParser(description="Build a manual longitudinal style profile from route logs.")
  parser.add_argument("routes", nargs="+", help="Routes, segment ranges, log files, or URLs accepted by LogReader")
  parser.add_argument("--qlog", action="store_true", help="Prefer qlogs instead of rlogs")
  parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
  parser.add_argument("--output", help="Write JSON summary to this path")
  parser.add_argument("--min-manual-moving", type=int, default=1200)
  parser.add_argument("--max-active-ratio", type=float, default=0.25)
  args = parser.parse_args()

  read_mode = ReadMode.QLOG if args.qlog else ReadMode.AUTO
  included_samples: list[ManualSample] = []
  route_profiles = []
  for route in args.routes:
    route_samples = extract_manual_samples(route, read_mode)
    route_profile = build_route_profile(route, route_samples, args.min_manual_moving, args.max_active_ratio)
    route_profiles.append(route_profile)
    if route_profile.include:
      included_samples.extend(route_samples)

  summary = summarize_manual_style(included_samples)
  payload = {"routes": [asdict(profile) for profile in route_profiles], "summary": asdict(summary)}
  if args.output:
    with open(args.output, "w") as f:
      json.dump(payload, f, indent=2)
      f.write("\n")
  print(json.dumps(payload, indent=2) if args.json else render_manual_style_summary(summary, route_profiles))


def extract_manual_samples(route: str, read_mode: ReadMode) -> list[ManualSample]:
  msgs = list(LogReader(route, default_mode=read_mode, sort_by_time=True))
  base_mono_time = int(getattr(msgs[0], "logMonoTime", 0)) if msgs else 0
  active = False
  lead_status = False
  lead_d_rel = None
  lead_v_rel = None
  samples: list[ManualSample] = []
  for msg in msgs:
    typ = msg_type(msg)
    payload = msg_payload(msg)
    if typ == "selfdriveState":
      active = bool(safe_get(payload, "active", False))
    elif typ == "radarState":
      lead = safe_get(payload, "leadOne")
      lead_status = bool(safe_get(lead, "status", False))
      lead_d_rel = _finite_or_none(safe_get(lead, "dRel"))
      lead_v_rel = _finite_or_none(safe_get(lead, "vRel"))
    elif typ == "carState":
      v_ego = _finite_or_none(safe_get(payload, "vEgo"))
      a_ego = _finite_or_none(safe_get(payload, "aEgo"))
      if v_ego is None or a_ego is None:
        continue
      samples.append(ManualSample(
        route=route,
        t=msg_time_s(msg, base_mono_time),
        v_ego=v_ego,
        a_ego=a_ego,
        active=active,
        gas_pressed=bool(safe_get(payload, "gasPressed", False)),
        brake_pressed=bool(safe_get(payload, "brakePressed", False)),
        lead_status=lead_status,
        lead_d_rel=lead_d_rel,
        lead_v_rel=lead_v_rel,
      ))
  return samples


def _finite_or_none(value: Any) -> float | None:
  if isinstance(value, int | float) and isfinite(float(value)):
    return float(value)
  return None


if __name__ == "__main__":
  main()
