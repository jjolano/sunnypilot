"""Pose-anchored CurveMemory: capture road curvature ahead with good vision, recall it through the
low-speed window, and only ever RAISE an under-curved vision toward the remembered corner (never
reduce a confident turn), vetoed by confident opposing vision. Default-off parity."""
from __future__ import annotations

from openpilot.sunnypilot.custom.lateral.demand.curve_memory import (
  MIN_CURVATURE,
  RECALL_MAX_V,
  CurveMemory,
  CurveMemoryInputs,
)

N = 30
DT = 0.01


def cm_in(v_ego, kappa_path, base, quality, enabled=True, lat_active=True) -> CurveMemoryInputs:
  # constant-curvature corner geometry: heading theta(s) = kappa*s, y = 0.5*kappa*s^2, x = s
  xs = [float(i) for i in range(N)]
  ys = [0.5 * kappa_path * i * i for i in range(N)]
  th = [kappa_path * i for i in range(N)]
  return CurveMemoryInputs(enabled=enabled, lat_active=lat_active, v_ego=v_ego, desired_curvature=base,
                           path_quality=quality, position_x=xs, position_y=ys, orientation_z=th, valid_path=True)


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
