"""Minimal nuScenes longitudinal trajectory importer for Drive Lab.

This module does NOT require the nuScenes devkit.  It reads a small JSON export
that contains an ego/lead time series and converts it into a Drive Lab Scenario.

Expected JSON schema::

  {
    "title": "<human-readable title> (optional)",
    "kind": "<scenario kind slug> (optional)",
    "duration": <float seconds, optional; defaults to last sample time>,
    "oracle_profile": "comfort" (optional),
    "provenance": {<source metadata>, optional},
    "samples": [
      {
        "t": <float seconds>,
        "ego_vx": <float m/s>,
        "lead_vx": <float m/s>,
        "d_rel": <float meters, optional>,
        "ego_x": <float meters, optional>,
        "lead_x": <float meters, optional>,
        "v_cruise": <float m/s, optional; defaults to ego speed>,
        "prob_lead": <float, optional; defaults to 1.0 when a lead is present>,
        "prob_throttle": <float, optional; defaults to 1.0>,
        "pitch": <float radians, optional; defaults to 0.0>
      },
      ...
    ]
  }

``d_rel`` may be supplied directly, or computed as ``lead_x - ego_x``.  The importer
normalizes time so that the first sample is at ``t == 0`` and uses the sample times
as maneuver breakpoints without resampling.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from openpilot.tools.drive_lab.longitudinal_route_extract import (
  LongitudinalRouteFrame,
  frames_to_maneuver_kwargs,
)
from openpilot.tools.drive_lab.longitudinal_scenarios import Scenario, _validate_mode


def generate_nuscenes_acc_scenarios(mode: str = "comfort", *, scenario_path: str) -> list[Scenario]:
  """Load a nuScenes longitudinal JSON export and return a single Scenario."""
  if scenario_path is None:
    raise TypeError("scenario_path is required")
  _validate_mode(mode)
  data = _load_json(Path(scenario_path))
  frames = _frames_from_data(data)
  if not frames:
    raise ValueError("expected at least one sample in the nuScenes export")
  return [_scenario_from_frames(data, frames, mode, str(scenario_path))]


def _load_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as handle:
    return json.load(handle)


def _frames_from_data(data: dict[str, Any]) -> tuple[LongitudinalRouteFrame, ...]:
  raw_samples = data.get("samples")
  if not isinstance(raw_samples, list):
    raise ValueError("expected 'samples' to be a list")

  raw_frames: list[LongitudinalRouteFrame] = []
  for sample in raw_samples:
    if not isinstance(sample, dict):
      raise ValueError("every sample must be a JSON object")
    if "t" not in sample:
      raise ValueError("every sample must include 't'")
    raw_frames.append(_frame_from_sample(sample))

  if not raw_frames:
    return ()

  t0 = raw_frames[0].source_t or 0.0
  return tuple(
    replace(frame, t=round((frame.source_t or 0.0) - t0, 6))
    for frame in raw_frames
  )


def _frame_from_sample(sample: dict[str, Any]) -> LongitudinalRouteFrame:
  v_ego = _extract_speed(sample, "ego")
  v_lead = _extract_lead_speed(sample)
  d_rel = _extract_d_rel(sample)
  lead_active = d_rel is not None
  v_cruise = _opt_float(sample, "v_cruise", v_ego)
  prob_lead = _opt_float(sample, "prob_lead", 1.0 if lead_active else 0.0)
  pitch = _opt_float(sample, "pitch", 0.0)
  prob_throttle = _opt_float(sample, "prob_throttle", 1.0)

  return LongitudinalRouteFrame(
    t=0.0,
    source_t=float(sample["t"]),
    v_ego=v_ego,
    v_cruise=v_cruise,
    pitch=pitch,
    lead_active=lead_active,
    d_rel=d_rel,
    v_lead=v_lead,
    prob_lead=prob_lead,
    prob_throttle=prob_throttle,
  )


def _extract_speed(sample: dict[str, Any], prefix: str) -> float:
  flat_keys = [f"{prefix}_vx", f"{prefix}_speed", f"{prefix}_v"]
  for key in flat_keys:
    if key in sample:
      return float(sample[key])
  nested = sample.get(prefix)
  if isinstance(nested, dict):
    for key in ("vx", "speed", "v"):
      if key in nested:
        return float(nested[key])
  raise ValueError(f"sample missing {prefix} speed (e.g. '{prefix}_vx')")


def _extract_lead_speed(sample: dict[str, Any]) -> float | None:
  try:
    return _extract_speed(sample, "lead")
  except ValueError:
    return None


def _extract_d_rel(sample: dict[str, Any]) -> float | None:
  if "d_rel" in sample and sample["d_rel"] is not None:
    return float(sample["d_rel"])
  ego_x = _extract_position(sample, "ego")
  lead_x = _extract_position(sample, "lead")
  if ego_x is not None and lead_x is not None:
    return lead_x - ego_x
  return None


def _extract_position(sample: dict[str, Any], prefix: str) -> float | None:
  flat_key = f"{prefix}_x"
  if flat_key in sample and sample[flat_key] is not None:
    return float(sample[flat_key])
  nested = sample.get(prefix)
  if isinstance(nested, dict) and "x" in nested and nested["x"] is not None:
    return float(nested["x"])
  return None


def _opt_float(sample: dict[str, Any], key: str, default: float) -> float:
  value = sample.get(key)
  if value is None:
    return default
  return float(value)


def _scenario_from_frames(
  data: dict[str, Any],
  frames: tuple[LongitudinalRouteFrame, ...],
  mode: str,
  path: str,
) -> Scenario:
  kwargs = frames_to_maneuver_kwargs(frames)
  title = str(data.get("title", Path(path).stem))
  kind = str(data.get("kind", "nuscenes_acc"))
  duration = float(data.get("duration", frames[-1].t))
  oracle_profile = str(data.get("oracle_profile", "comfort"))
  provenance = dict(data.get("provenance", {}))
  provenance.setdefault("source", "nuscenes")
  provenance.setdefault("path", path)
  return Scenario(
    mode,
    kind,
    title,
    duration,
    kwargs,
    oracle_profile=oracle_profile,
    provenance=provenance,
  )
