#!/usr/bin/env python3
"""Extract car-following segments from OpenACC CSV exports into LongitudinalProfile JSON."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openpilot.tools.drive_lab.log_profile import LongitudinalProfile, ProfileRange, save_profile


def build_openacc_profile(csv_path: str | Path, *, source: str | None = None) -> LongitudinalProfile:
  path = Path(csv_path)
  ego_speeds: list[float] = []
  lead_gaps: list[float] = []
  closing_speeds: list[float] = []
  lead_decels: list[float] = []
  prev_v_lead: float | None = None
  prev_t: float | None = None

  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      v_ego = _float_col(row, ("v_ego", "V_ego", "speed_ego", "SpeedEgo"))
      v_lead = _float_col(row, ("v_lead", "V_lead", "speed_lead", "SpeedLead"))
      gap = _float_col(row, ("gap", "Gap", "distance", "Distance", "s", "S"))
      t = _float_col(row, ("time", "Time", "t", "timestamp"))
      if v_ego is None:
        continue
      ego_speeds.append(v_ego)
      if gap is not None:
        lead_gaps.append(gap)
      if v_lead is not None:
        if v_ego > v_lead:
          closing_speeds.append(v_ego - v_lead)
        if prev_v_lead is not None and prev_t is not None and t is not None:
          dt = t - prev_t
          if dt > 1e-3:
            accel = (v_lead - prev_v_lead) / dt
            if accel < 0.0:
              lead_decels.append(-accel)
        prev_v_lead = v_lead
        prev_t = t

  def _range(values: list[float], fallback: tuple[float, float]) -> ProfileRange:
    if len(values) < 5:
      return ProfileRange(*fallback)
    values = sorted(values)
    lo = values[int(0.15 * (len(values) - 1))]
    hi = values[int(0.85 * (len(values) - 1))]
    return ProfileRange(lo, hi)

  label = source or path.stem
  return LongitudinalProfile(
    source=f"openacc:{label}",
    sample_count=len(ego_speeds),
    ego_speed=_range(ego_speeds, (8.0, 24.0)),
    cruise_speed=_range(ego_speeds, (5.0, 15.0)),
    lead_gap=_range(lead_gaps, (25.0, 70.0)),
    closing_speed=_range(closing_speeds, (0.0, 4.0)),
    lead_decel=_range(lead_decels, (1.5, 3.5)),
    stopped_lead_gap=ProfileRange(4.5, 8.0),
    lead_pullaway_speed=ProfileRange(1.0, 3.5),
  )


def load_openacc_profile(path: str | Path) -> LongitudinalProfile:
  return build_openacc_profile(path)


def _float_col(row: dict[str, str], names: tuple[str, ...]) -> float | None:
  for name in names:
    if name in row and row[name] not in ("", "NA", "nan"):
      try:
        return float(row[name])
      except ValueError:
        continue
  return None


def main() -> None:
  parser = argparse.ArgumentParser(description="Build a LongitudinalProfile from an OpenACC CSV export.")
  parser.add_argument("csv_path", help="OpenACC or compatible car-following CSV")
  parser.add_argument("--output", required=True, help="Write profile JSON here")
  parser.add_argument("--source", help="Profile source label")
  args = parser.parse_args()

  profile = build_openacc_profile(args.csv_path, source=args.source)
  save_profile(profile, args.output)
  print(json.dumps(profile.to_dict(), indent=2))


if __name__ == "__main__":
  main()
