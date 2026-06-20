"""Pose-anchored CurveMemory: capture road curvature ahead with good vision, recall it through the
low-speed window, and only ever RAISE an under-curved vision toward the remembered corner (never
reduce a confident turn), vetoed by confident opposing vision. Default-off parity."""
from __future__ import annotations

from openpilot.sunnypilot.custom.lateral.demand.curve_memory import (
  CAPTURE_MIN_V,
  MIN_CURVATURE,
  RECALL_MAX_V,
  CurveMemory,
  CurveMemoryInputs,
)

N = 30
DT = 0.01


def cm_in(v_ego, kappa_path, base, quality, enabled=True, lat_active=True, steering_pressed: bool | None = False,
          path_reason="ok", lane_change_active=False, path_gated=False, valid_path=True) -> CurveMemoryInputs:
  # constant-curvature corner geometry: heading theta(s) = kappa*s, y = 0.5*kappa*s^2, x = s
  xs = [float(i) for i in range(N)]
  ys = [0.5 * kappa_path * i * i for i in range(N)]
  th = [kappa_path * i for i in range(N)]
  return CurveMemoryInputs(enabled=enabled, lat_active=lat_active, v_ego=v_ego, desired_curvature=base,
                           path_quality=quality, path_reason=path_reason, path_gated=path_gated,
                           steering_pressed=steering_pressed, lane_change_active=lane_change_active,
                           position_x=xs, position_y=ys, orientation_z=th, valid_path=valid_path)


def _drive_corner(cm: CurveMemory, kappa=0.02, v=5.0, frames=60) -> None:
  for _ in range(frames):                     # approach at good speed + confident vision -> capture
    cm.update(cm_in(v, kappa, base=kappa, quality=1.0), DT)


def test_recall_raises_collapsed_vision_toward_remembered_corner():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  # vision collapses at low speed (the amnesia window): base ~0, degraded quality
  r = cm.update(cm_in(0.5, kappa_path=0.02, base=0.0, quality=0.3), DT)
  assert r.active and r.source == "memory"
  assert 0.01 < r.desired_curvature <= 0.02     # raised toward the remembered corner, never past it


def test_memory_never_reduces_a_confident_turn():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  # vision already turning MORE than the remembered corner -> memory must not pull it down
  r = cm.update(cm_in(0.5, kappa_path=0.02, base=0.025, quality=0.3), DT)
  assert not r.active and r.desired_curvature == 0.025


def test_confident_opposing_vision_vetoes_memory():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  r = cm.update(cm_in(0.5, kappa_path=0.02, base=-0.02, quality=0.95), DT)   # confident, opposite sign
  assert r.source == "vetoed" and r.desired_curvature == -0.02


def test_no_recall_above_speed_window():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  r = cm.update(cm_in(RECALL_MAX_V + 2.0, kappa_path=0.02, base=0.0, quality=0.3), DT)
  assert not r.active and r.desired_curvature == 0.0     # vision authority above the degraded window


def test_disabled_is_passthrough():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  r = cm.update(cm_in(0.5, kappa_path=0.02, base=0.0, quality=0.3, enabled=False), DT)
  assert r.source == "disabled" and r.desired_curvature == 0.0


def test_straight_road_is_not_remembered():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.001, frames=60)               # below MIN_CURVATURE
  r = cm.update(cm_in(0.5, kappa_path=0.001, base=0.0, quality=0.3), DT)
  assert not r.active and r.desired_curvature == 0.0
  assert MIN_CURVATURE > 0.001                             # guard the premise


def test_only_captures_from_fast_confident_frames():
  # slow throughout (intersection from a crawl): no good-vision capture -> nothing to recall
  cm = CurveMemory()
  for _ in range(60):
    cm.update(cm_in(1.0, kappa_path=0.02, base=0.005, quality=0.5), DT)   # v<CAPTURE_MIN_V, low quality
  r = cm.update(cm_in(0.5, kappa_path=0.02, base=0.0, quality=0.3), DT)
  assert not r.active                                      # honest limit: can't remember what was never seen well


def test_no_capture_or_recall_while_steering_pressed():
  cm = CurveMemory()
  for _ in range(30):
    cm.update(cm_in(5.0, kappa_path=0.02, base=0.02, quality=1.0, steering_pressed=True), DT)
  r = cm.update(cm_in(0.5, kappa_path=0.02, base=0.0, quality=0.3, steering_pressed=True), DT)
  assert not r.active and r.source == "driver_override"


def test_unknown_driver_state_inhibits_recall():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  r = cm.update(cm_in(0.5, kappa_path=0.02, base=0.0, quality=0.3, steering_pressed=None), DT)
  assert not r.active and r.source == "driver_override"


def test_post_release_inhibit_blocks_recall():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  cm.update(cm_in(0.5, 0.02, 0.0, 0.3, steering_pressed=True), DT)
  r = cm.update(cm_in(0.5, 0.02, 0.0, 0.3, steering_pressed=False), DT)
  assert not r.active and r.source == "driver_recall_inhibit"


def test_lane_change_clears_and_suppresses_memory():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  r = cm.update(cm_in(0.5, 0.02, 0.0, 0.3, lane_change_active=True), DT)
  assert not r.active and r.source == "lane_change"
  r2 = cm.update(cm_in(0.5, 0.02, 0.0, 0.3, lane_change_active=False), DT)
  assert not r2.active


def test_invalid_path_hard_weight_only_with_trusted_memory():
  cm = CurveMemory()
  _drive_corner(cm, kappa=0.02)
  r = cm.update(cm_in(0.5, 0.02, 0.0, 0.2, path_reason="invalid_path"), DT)
  assert r.active and r.desired_curvature > 0.01


def test_hard_degraded_capture_is_blocked():
  cm = CurveMemory()
  for _ in range(30):
    cm.update(cm_in(5.0, 0.02, 0.02, 1.0, path_reason="invalid_path"), DT)
  r = cm.update(cm_in(0.5, 0.02, 0.0, 0.3), DT)
  assert not r.active


def test_capture_speed_gate_blocks_capture_below_min_speed():
  cm = CurveMemory()
  r = None
  for _ in range(100):
    r = cm.update(cm_in(CAPTURE_MIN_V - 1.0, kappa_path=0.02, base=0.02, quality=1.0), DT)
  assert r is not None and r.samples == 0
  r2 = cm.update(cm_in(0.5, kappa_path=0.02, base=0.0, quality=0.3), DT)
  assert not r2.active and r2.source == "vision"


def test_capture_speed_gate_allows_capture_at_min_speed():
  cm = CurveMemory()
  r = None
  for _ in range(60):
    r = cm.update(cm_in(CAPTURE_MIN_V, kappa_path=0.02, base=0.02, quality=1.0), DT)
  assert r is not None and r.samples > 0
  r2 = cm.update(cm_in(0.5, kappa_path=0.02, base=0.0, quality=0.3), DT)
  assert r2.active and r2.source == "memory"


def test_capture_speed_gate_is_fail_closed_for_nonfinite_speed():
  cm = CurveMemory()
  for v in (float("nan"), float("inf"), float("-inf")):
    r = None
    for _ in range(30):
      r = cm.update(cm_in(v, kappa_path=0.02, base=0.02, quality=1.0), DT)
    assert r is not None and r.samples == 0
