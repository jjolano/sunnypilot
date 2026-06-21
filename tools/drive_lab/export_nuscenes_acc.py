#!/usr/bin/env python3
"""Export nuScenes mini/raw metadata as Drive Lab longitudinal JSON scenarios.

Reads the official nuScenes metadata JSON tables directly (no devkit required)
from ``DATAROOT/<version>/`` and writes stable lead segments that can be consumed
by ``tools/drive_lab/nuscenes_acc.py``.

Example::

  python tools/drive_lab/export_nuscenes_acc.py /data/nuscenes \
    --version v1.0-mini --output-dir /tmp/nuscenes_acc
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterator

# Official nuScenes tables (v1.0-mini) that this exporter reads.
_TABLE_NAMES = (
  "scene",
  "sample",
  "sample_data",
  "ego_pose",
  "calibrated_sensor",
  "sensor",
  "sample_annotation",
  "instance",
  "category",
)

_GAP_THRESHOLD_S = 0.75


def iter_segments(
  dataroot: str | Path,
  version: str = "v1.0-mini",
  *,
  min_segment_samples: int = 5,
  max_lead_lateral_m: float = 8.0,
  min_lead_distance_m: float = 5.0,
  max_lead_distance_m: float = 120.0,
  comfort_min_ttc_s: float = 2.5,
) -> Iterator[dict[str, Any]]:
  """Yield scenario dictionaries, one per stable lead segment."""
  tables = _load_tables(dataroot, version)
  indexes = _build_indexes(tables)

  for scene in _ordered_scenes(tables):
    yield from _process_scene(
      scene,
      tables,
      indexes,
      version=version,
      min_segment_samples=min_segment_samples,
      max_lead_lateral_m=max_lead_lateral_m,
      min_lead_distance_m=min_lead_distance_m,
      max_lead_distance_m=max_lead_distance_m,
      comfort_min_ttc_s=comfort_min_ttc_s,
    )


def export_nuscenes_acc(
  dataroot: str | Path,
  version: str,
  output_dir: str | Path,
  *,
  min_segment_samples: int = 5,
  max_lead_lateral_m: float = 8.0,
  min_lead_distance_m: float = 5.0,
  max_lead_distance_m: float = 120.0,
  comfort_min_ttc_s: float = 2.5,
) -> int:
  """Write all stable lead segments to ``output_dir``.

  Returns the number of segments written.
  """
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  counts_by_scene: dict[str, int] = defaultdict(int)
  written = 0

  for segment in iter_segments(
    dataroot,
    version,
    min_segment_samples=min_segment_samples,
    max_lead_lateral_m=max_lead_lateral_m,
    min_lead_distance_m=min_lead_distance_m,
    max_lead_distance_m=max_lead_distance_m,
    comfort_min_ttc_s=comfort_min_ttc_s,
  ):
    scene_name = segment["provenance"]["scene_name"]
    idx = counts_by_scene[scene_name]
    counts_by_scene[scene_name] += 1
    out_file = output_path / f"{scene_name}_lead_{idx}.json"
    with out_file.open("w", encoding="utf-8") as handle:
      json.dump(segment, handle, indent=2)
    written += 1

  return written


def _load_tables(dataroot: str | Path, version: str) -> dict[str, dict[str, Any]]:
  base = Path(dataroot) / version
  if not base.is_dir():
    raise FileNotFoundError(f"nuScenes metadata directory not found: {base}")

  tables: dict[str, dict[str, Any]] = {}
  for name in _TABLE_NAMES:
    path = base / f"{name}.json"
    if not path.is_file():
      raise FileNotFoundError(f"missing nuScenes table: {path}")
    with path.open(encoding="utf-8") as handle:
      records = json.load(handle)
    if not isinstance(records, list):
      raise ValueError(f"expected a list of records in {path}")
    tables[name] = {record["token"]: record for record in records}

  return tables


def _build_indexes(tables: dict[str, dict[str, Any]]) -> dict[str, Any]:
  calibration_to_channel: dict[str, str] = {}
  for cs_token, cs in tables["calibrated_sensor"].items():
    sensor = tables["sensor"][cs["sensor_token"]]
    calibration_to_channel[cs_token] = sensor.get("channel", "")

  lidar_sample_data: dict[str, Any] = {}
  for sd in tables["sample_data"].values():
    if sd.get("is_key_frame") and calibration_to_channel.get(sd["calibrated_sensor_token"]) == "LIDAR_TOP":
      lidar_sample_data[sd["sample_token"]] = sd

  annotations_by_sample: dict[str, list[str]] = defaultdict(list)
  for ann_token, ann in tables["sample_annotation"].items():
    annotations_by_sample[ann["sample_token"]].append(ann_token)

  instance_category: dict[str, str] = {}
  for inst_token, inst in tables["instance"].items():
    instance_category[inst_token] = inst["category_token"]

  return {
    "calibration_to_channel": calibration_to_channel,
    "lidar_sample_data": lidar_sample_data,
    "annotations_by_sample": dict(annotations_by_sample),
    "instance_category": instance_category,
  }


def _ordered_scenes(tables: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
  return sorted(tables["scene"].values(), key=lambda s: s.get("name", s["token"]))


def _process_scene(
  scene: dict[str, Any],
  tables: dict[str, dict[str, Any]],
  indexes: dict[str, Any],
  *,
  version: str,
  min_segment_samples: int,
  max_lead_lateral_m: float,
  min_lead_distance_m: float,
  max_lead_distance_m: float,
  comfort_min_ttc_s: float,
) -> Iterator[dict[str, Any]]:
  sample_tokens = _scene_sample_tokens(scene, tables)
  if len(sample_tokens) < 2:
    return

  poses = []
  positions: list[tuple[float, float]] = []
  yaws: list[float] = []
  timestamps_s: list[float] = []
  for sample_token in sample_tokens:
    sd = indexes["lidar_sample_data"].get(sample_token)
    if sd is None:
      return
    pose = tables["ego_pose"][sd["ego_pose_token"]]
    poses.append(pose)
    tx, ty, _ = pose["translation"]
    positions.append((float(tx), float(ty)))
    yaws.append(_yaw_from_quaternion(pose["rotation"]))
    timestamps_s.append(float(tables["sample"][sample_token]["timestamp"]) / 1e6)

  leads: list[dict[str, Any] | None] = []
  lead_distances: list[float] = []
  for i, sample_token in enumerate(sample_tokens):
    lead = _choose_lead(
      sample_token,
      poses[i],
      yaws[i],
      tables,
      indexes,
      min_lead_distance_m=min_lead_distance_m,
      max_lead_distance_m=max_lead_distance_m,
      max_lead_lateral_m=max_lead_lateral_m,
    )
    leads.append(lead)
    if lead is None:
      lead_distances.append(0.0)
    else:
      dx = lead["translation"][0] - poses[i]["translation"][0]
      dy = lead["translation"][1] - poses[i]["translation"][1]
      x_ego, _ = _to_ego_frame(dx, dy, yaws[i])
      lead_distances.append(x_ego)

  segment_indices = _split_segments(sample_tokens, timestamps_s, leads)

  for seg in segment_indices:
    if len(seg) < min_segment_samples:
      continue
    yield _build_segment(
      scene,
      version,
      sample_tokens,
      poses,
      positions,
      yaws,
      timestamps_s,
      leads,
      lead_distances,
      seg,
      comfort_min_ttc_s=comfort_min_ttc_s,
    )


def _scene_sample_tokens(scene: dict[str, Any], tables: dict[str, dict[str, Any]]) -> list[str]:
  tokens: list[str] = []
  token = scene.get("first_sample_token")
  last = scene.get("last_sample_token")
  while token:
    tokens.append(token)
    if token == last:
      break
    sample = tables["sample"].get(token)
    if sample is None:
      break
    token = sample.get("next")
  return tokens


def _choose_lead(
  sample_token: str,
  pose: dict[str, Any],
  yaw: float,
  tables: dict[str, dict[str, Any]],
  indexes: dict[str, Any],
  *,
  min_lead_distance_m: float,
  max_lead_distance_m: float,
  max_lead_lateral_m: float,
) -> dict[str, Any] | None:
  best_ann: dict[str, Any] | None = None
  best_x: float | None = None
  px, py, _ = pose["translation"]

  for ann_token in indexes["annotations_by_sample"].get(sample_token, []):
    ann = tables["sample_annotation"][ann_token]
    cat = tables["category"].get(indexes["instance_category"].get(ann["instance_token"]))
    if cat is None or not str(cat.get("name", "")).startswith("vehicle."):
      continue
    ax, ay, _ = ann["translation"]
    dx = ax - px
    dy = ay - py
    x_ego, y_ego = _to_ego_frame(dx, dy, yaw)
    if x_ego < min_lead_distance_m or x_ego > max_lead_distance_m or abs(y_ego) > max_lead_lateral_m:
      continue
    if best_x is None or x_ego < best_x:
      best_ann = ann
      best_x = x_ego

  return best_ann


def _split_segments(
  sample_tokens: list[str],
  timestamps_s: list[float],
  leads: list[dict[str, Any] | None],
) -> list[list[int]]:
  segments: list[list[int]] = []
  current: list[int] = []
  prev_instance: str | None = None
  prev_time: float | None = None

  for i, lead in enumerate(leads):
    instance = lead["instance_token"] if lead is not None else None
    gap = prev_time is not None and (timestamps_s[i] - prev_time) > _GAP_THRESHOLD_S

    if instance is None or instance != prev_instance or gap:
      if current:
        segments.append(current)
      current = [i] if instance is not None else []
      prev_instance = instance
    else:
      current.append(i)

    prev_time = timestamps_s[i]

  if current:
    segments.append(current)

  return [seg for seg in segments if seg]


def _build_segment(
  scene: dict[str, Any],
  version: str,
  sample_tokens: list[str],
  poses: list[dict[str, Any]],
  positions: list[tuple[float, float]],
  yaws: list[float],
  timestamps_s: list[float],
  leads: list[dict[str, Any] | None],
  lead_distances: list[float],
  seg: list[int],
  *,
  comfort_min_ttc_s: float,
) -> dict[str, Any]:
  first = seg[0]
  t0 = timestamps_s[first]

  first_lead = leads[first]
  assert first_lead is not None
  lead_instance = first_lead["instance_token"]
  seg_positions: list[tuple[float, float] | None] = []
  seg_timestamps: list[float] = []
  for i in seg:
    lead = leads[i]
    assert lead is not None
    ax, ay, _ = lead["translation"]
    seg_positions.append((float(ax), float(ay)))
    seg_timestamps.append(timestamps_s[i])

  ego_speeds = [_finite_diff_speed(i, positions, timestamps_s) for i in seg]
  lead_speeds = [_finite_diff_speed(k, seg_positions, seg_timestamps) for k in range(len(seg))]

  samples: list[dict[str, Any]] = []
  for k, i in enumerate(seg):
    lead = leads[i]
    assert lead is not None
    samples.append({
      "t": round(timestamps_s[i] - t0, 6),
      "ego_vx": round(ego_speeds[k], 3),
      "lead_vx": round(lead_speeds[k], 3),
      "d_rel": round(lead_distances[i], 3),
      "prob_lead": 1.0,
      "sample_token": sample_tokens[i],
      "annotation_token": lead["token"],
      "ego_pose_token": poses[i]["token"],
    })

  d_rel0 = samples[0]["d_rel"]
  v_ego0 = samples[0]["ego_vx"]
  v_lead0 = samples[0]["lead_vx"]
  closing = max(0.0, v_ego0 - v_lead0)
  ttc = d_rel0 / closing if closing > 0.0 else float("inf")
  oracle_profile = "safety" if (closing > 0.0 and ttc < comfort_min_ttc_s) else "comfort"

  provenance = {
    "source": "nuscenes",
    "version": version,
    "scene_name": str(scene.get("name", scene["token"])),
    "scene_description": str(scene.get("description", "")),
    "lead_instance_token": lead_instance,
  }
  if oracle_profile == "safety":
    provenance["classification"] = "urgent"
    provenance["comfort_min_ttc_s"] = comfort_min_ttc_s

  return {
    "title": f"nuscenes {provenance['scene_name']} lead_{lead_instance[:6]}",
    "kind": "nuscenes_exported_lead",
    "duration": round(timestamps_s[seg[-1]] - t0, 3),
    "oracle_profile": oracle_profile,
    "provenance": provenance,
    "samples": samples,
  }


def _finite_diff_speed(
  i: int,
  positions: Sequence[tuple[float, float] | None],
  timestamps_s: list[float],
) -> float:
  pos_i = positions[i]
  if pos_i is None:
    return 0.0

  def take(j: int) -> tuple[float, float] | None:
    return positions[j] if 0 <= j < len(positions) else None

  prev_pos = take(i - 1)
  next_pos = take(i + 1)
  if prev_pos is not None and next_pos is not None:
    dt = timestamps_s[i + 1] - timestamps_s[i - 1]
    if dt > 0.0:
      return math.hypot(next_pos[0] - prev_pos[0], next_pos[1] - prev_pos[1]) / dt
  if next_pos is not None:
    dt = timestamps_s[i + 1] - timestamps_s[i]
    if dt > 0.0:
      return math.hypot(next_pos[0] - pos_i[0], next_pos[1] - pos_i[1]) / dt
  if prev_pos is not None:
    dt = timestamps_s[i] - timestamps_s[i - 1]
    if dt > 0.0:
      return math.hypot(pos_i[0] - prev_pos[0], pos_i[1] - prev_pos[1]) / dt
  return 0.0


def _yaw_from_quaternion(q: list[float]) -> float:
  """Return z-axis rotation (heading) of a quaternion [w, x, y, z]."""
  w, x, y, z = q
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _to_ego_frame(dx: float, dy: float, yaw: float) -> tuple[float, float]:
  """Rotate a global delta into the ego frame (x-forward, y-left)."""
  c = math.cos(yaw)
  s = math.sin(yaw)
  return dx * c + dy * s, -dx * s + dy * c


def main() -> None:
  parser = argparse.ArgumentParser(description="Export nuScenes lead segments for Drive Lab.")
  parser.add_argument("dataroot", help="Path to the nuScenes dataset root")
  parser.add_argument("--version", default="v1.0-mini", help="NuScenes version directory name (default: v1.0-mini)")
  parser.add_argument("--output-dir", required=True, help="Directory to write JSON scenario files")
  parser.add_argument("--min-segment-samples", type=int, default=5, help="Minimum samples per exported segment")
  parser.add_argument("--max-lead-lateral-m", type=float, default=8.0, help="Maximum absolute lateral offset of a lead")
  parser.add_argument("--min-lead-distance-m", type=float, default=5.0, help="Minimum longitudinal distance to a lead")
  parser.add_argument("--max-lead-distance-m", type=float, default=120.0, help="Maximum longitudinal distance to a lead")
  parser.add_argument("--comfort-min-ttc-s", type=float, default=2.5, help="TTC threshold below which segment is marked safety")
  args = parser.parse_args()

  written = export_nuscenes_acc(
    args.dataroot,
    args.version,
    args.output_dir,
    min_segment_samples=args.min_segment_samples,
    max_lead_lateral_m=args.max_lead_lateral_m,
    min_lead_distance_m=args.min_lead_distance_m,
    max_lead_distance_m=args.max_lead_distance_m,
    comfort_min_ttc_s=args.comfort_min_ttc_s,
  )
  print(f"Wrote {written} nuScenes lead segment(s) to {args.output_dir}")


if __name__ == "__main__":
  main()
