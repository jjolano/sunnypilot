import json
from pathlib import Path

import pytest

from openpilot.tools.drive_lab.export_nuscenes_acc import (
  export_nuscenes_acc,
  iter_segments,
)
from openpilot.tools.drive_lab.nuscenes_acc import generate_nuscenes_acc_scenarios


def _write_table(root: Path, name: str, records: list[dict]) -> None:
  (root / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")


def _identity_quat() -> list[float]:
  return [1.0, 0.0, 0.0, 0.0]


def _default_sensor_tables(root: Path) -> str:
  sensor_token = "sensor_lidar_top"
  calibrated_token = "cs_lidar_top"
  _write_table(root, "sensor", [{
    "token": sensor_token,
    "channel": "LIDAR_TOP",
    "modality": "lidar",
  }])
  _write_table(root, "calibrated_sensor", [{
    "token": calibrated_token,
    "sensor_token": sensor_token,
    "translation": [0.0, 0.0, 0.0],
    "rotation": _identity_quat(),
  }])
  return calibrated_token


def _write_categories_and_instances(root: Path, instance_tokens: set[str], category_name: str = "vehicle.car") -> None:
  category_token = f"category_{category_name.replace('.', '_')}"
  _write_table(root, "category", [{
    "token": category_token,
    "name": category_name,
    "description": "",
    "index": 0,
  }])
  _write_table(root, "instance", [
    {
      "token": inst,
      "category_token": category_token,
      "nbr_annotations": 0,
      "first_annotation_token": "",
      "last_annotation_token": "",
    }
    for inst in sorted(instance_tokens)
  ])


def _make_scene(
  tmp_path: Path,
  scene_name: str,
  scene_description: str,
  lead_positions: list[tuple[str, tuple[float, float, float]]],
  ego_positions: list[tuple[float, float, float]],
  timestamps_us: list[int],
  version: str = "v1.0-mini",
) -> Path:
  root = tmp_path / "nuscenes"
  version_dir = root / version
  version_dir.mkdir(parents=True)

  cs_token = _default_sensor_tables(version_dir)

  num_samples = len(timestamps_us)
  assert num_samples == len(ego_positions)
  assert num_samples == len(lead_positions)

  scene_token = f"scene_{scene_name}"
  sample_tokens = [f"sample_{scene_name}_{i}" for i in range(num_samples)]
  sample_data_tokens = [f"sd_{scene_name}_{i}" for i in range(num_samples)]
  ego_pose_tokens = [f"ego_{scene_name}_{i}" for i in range(num_samples)]
  annotation_tokens = [f"ann_{scene_name}_{i}" for i in range(num_samples)]

  samples: list[dict] = []
  sample_data: list[dict] = []
  ego_poses: list[dict] = []
  annotations: list[dict] = []
  for i in range(num_samples):
    samples.append({
      "token": sample_tokens[i],
      "scene_token": scene_token,
      "timestamp": timestamps_us[i],
      "next": sample_tokens[i + 1] if i + 1 < num_samples else "",
      "prev": sample_tokens[i - 1] if i > 0 else "",
      "data": {},
      "anns": [annotation_tokens[i]],
    })
    sample_data.append({
      "token": sample_data_tokens[i],
      "sample_token": sample_tokens[i],
      "ego_pose_token": ego_pose_tokens[i],
      "calibrated_sensor_token": cs_token,
      "filename": "",
      "fileformat": ".pcd.bin",
      "width": 0,
      "height": 0,
      "timestamp": timestamps_us[i],
      "is_key_frame": True,
      "prev": sample_data_tokens[i - 1] if i > 0 else "",
      "next": sample_data_tokens[i + 1] if i + 1 < num_samples else "",
    })
    ego_poses.append({
      "token": ego_pose_tokens[i],
      "timestamp": timestamps_us[i],
      "rotation": _identity_quat(),
      "translation": list(ego_positions[i]),
      "prev": ego_pose_tokens[i - 1] if i > 0 else "",
      "next": ego_pose_tokens[i + 1] if i + 1 < num_samples else "",
    })
    instance_token, ann_translation = lead_positions[i]
    annotations.append({
      "token": annotation_tokens[i],
      "sample_token": sample_tokens[i],
      "instance_token": instance_token,
      "category_token": f"category_vehicle_car",
      "attribute_tokens": [],
      "visibility_token": "",
      "translation": list(ann_translation),
      "size": [1.0, 1.0, 1.0],
      "rotation": _identity_quat(),
      "prev": annotation_tokens[i - 1] if i > 0 else "",
      "next": annotation_tokens[i + 1] if i + 1 < num_samples else "",
      "num_lidar_pts": 1,
      "num_radar_pts": 0,
    })

  _write_categories_and_instances(
    version_dir,
    {instance for instance, _ in lead_positions},
  )

  _write_table(version_dir, "scene", [{
    "token": scene_token,
    "name": scene_name,
    "description": scene_description,
    "first_sample_token": sample_tokens[0],
    "last_sample_token": sample_tokens[-1],
  }])
  _write_table(version_dir, "sample", samples)
  _write_table(version_dir, "sample_data", sample_data)
  _write_table(version_dir, "ego_pose", ego_poses)
  _write_table(version_dir, "sample_annotation", annotations)

  return root


def test_segments_split_on_instance_switch(tmp_path: Path):
  root = _make_scene(
    tmp_path,
    scene_name="switch",
    scene_description="two lead instances",
    ego_positions=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0), (15.0, 0.0, 0.0)],
    lead_positions=[
      ("lead_a", (30.0, 0.0, 0.0)),
      ("lead_a", (35.0, 0.0, 0.0)),
      ("lead_b", (50.0, 0.0, 0.0)),
      ("lead_b", (55.0, 0.0, 0.0)),
    ],
    timestamps_us=[0, 500_000, 1_000_000, 1_500_000],
  )

  segments = list(iter_segments(root, min_segment_samples=1))
  assert len(segments) == 2
  assert segments[0]["provenance"]["lead_instance_token"] == "lead_a"
  assert segments[1]["provenance"]["lead_instance_token"] == "lead_b"
  assert len(segments[0]["samples"]) == 2
  assert len(segments[1]["samples"]) == 2


def test_urgent_segment_marked_safety(tmp_path: Path):
  root = _make_scene(
    tmp_path,
    scene_name="urgent",
    scene_description="closing fast",
    ego_positions=[(0.0, 0.0, 0.0), (15.0, 0.0, 0.0), (30.0, 0.0, 0.0)],
    lead_positions=[
      ("lead_a", (20.0, 0.0, 0.0)),
      ("lead_a", (30.0, 0.0, 0.0)),
      ("lead_a", (40.0, 0.0, 0.0)),
    ],
    timestamps_us=[0, 500_000, 1_000_000],
  )

  segments = list(iter_segments(root, min_segment_samples=1))
  assert len(segments) == 1
  assert segments[0]["oracle_profile"] == "safety"
  assert segments[0]["provenance"].get("classification") == "urgent"


def test_comfort_segment_marked_comfort(tmp_path: Path):
  root = _make_scene(
    tmp_path,
    scene_name="comfort",
    scene_description="steady following",
    ego_positions=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
    lead_positions=[
      ("lead_a", (50.0, 0.0, 0.0)),
      ("lead_a", (55.0, 0.0, 0.0)),
      ("lead_a", (60.0, 0.0, 0.0)),
    ],
    timestamps_us=[0, 500_000, 1_000_000],
  )

  segments = list(iter_segments(root, min_segment_samples=1))
  assert len(segments) == 1
  assert segments[0]["oracle_profile"] == "comfort"
  assert "classification" not in segments[0]["provenance"]


def test_exported_json_loadable_by_importer(tmp_path: Path):
  out_dir = tmp_path / "out"
  root = _make_scene(
    tmp_path,
    scene_name="roundtrip",
    scene_description="round trip test",
    ego_positions=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
    lead_positions=[
      ("lead_a", (50.0, 0.0, 0.0)),
      ("lead_a", (55.0, 0.0, 0.0)),
      ("lead_a", (60.0, 0.0, 0.0)),
    ],
    timestamps_us=[0, 500_000, 1_000_000],
    version="v1.0-mini",
  )

  written = export_nuscenes_acc(
    root,
    "v1.0-mini",
    out_dir,
    min_segment_samples=1,
  )
  assert written == 1

  out_files = sorted(out_dir.glob("*.json"))
  assert len(out_files) == 1
  scenarios = generate_nuscenes_acc_scenarios("comfort", scenario_path=str(out_files[0]))
  assert len(scenarios) == 1
  scenario = scenarios[0]
  kwargs = scenario.kwargs
  assert kwargs["lead_relevancy"] is True
  assert kwargs["initial_distance_lead"] == 50.0
  assert scenario.oracle_profile == "comfort"


def test_min_segment_samples_filters_short_segments(tmp_path: Path):
  root = _make_scene(
    tmp_path,
    scene_name="short",
    scene_description="single sample segment",
    ego_positions=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
    lead_positions=[
      ("lead_a", (30.0, 0.0, 0.0)),
      ("lead_b", (50.0, 0.0, 0.0)),
      ("lead_b", (55.0, 0.0, 0.0)),
    ],
    timestamps_us=[0, 500_000, 1_000_000],
  )

  segments = list(iter_segments(root, min_segment_samples=2))
  assert len(segments) == 1
  assert segments[0]["provenance"]["lead_instance_token"] == "lead_b"
  assert len(segments[0]["samples"]) == 2
