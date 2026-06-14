"""Curve-memory tests for ModelPathProcessor: hold a decaying corner prior across a brief
standstill (stopped mid-corner) and re-seed the retained-curve fallback on launch, so the car
resumes the corner instead of starting cold on conservative low-speed vision. Opt-in; reuses the
retained-curve sign/closeness safety checks. Default-off parity is asserted too.
"""
from __future__ import annotations

from openpilot.sunnypilot.custom.lateral.demand.model_path_processor import (
  STANDSTILL_PRIOR_MAX_SECONDS,
  ModelPathProcessor,
  ModelPathProcessorInputs,
)

N = 33  # >= PATH_VALID_MIN_LEN
CORNER = 0.02      # a real corner (|k| >= LOW_SPEED_CURVE_RETENTION_MIN_CURVATURE = 0.008)
DT = 0.01


def mp_inputs(*, lat_active=True, v_ego=8.0, desired=CORNER, measured=CORNER, prev=CORNER,
              y_std=0.1, lane_prob=0.9, curve_memory_enabled=False) -> ModelPathProcessorInputs:
  xs = [float(x) for x in range(N)]
  ys = [0.5 * desired * x * x for x in range(N)]
  yaw = [desired * x for x in range(N)]
  yaw_rate = [desired * v_ego] * N
  return ModelPathProcessorInputs(
    lat_active=lat_active, v_ego=v_ego, desired_curvature=desired, measured_curvature=measured,
    previous_desired_curvature=prev,
    position_x=xs, position_y=ys, position_y_std=[y_std] * N,
    orientation_z=yaw, orientation_rate_z=yaw_rate, lane_line_probs=[lane_prob] * 4,
    curve_memory_enabled=curve_memory_enabled,
  )


def _drive_corner_then_stop(p: ModelPathProcessor, curve_memory: bool, stop_frames: int = 20) -> None:
  for _ in range(5):                                          # establish the corner while moving
    p.update(mp_inputs(lat_active=True, curve_memory_enabled=curve_memory))
  # active -> inactive transition: previous curvature still holds the corner
  p.update(mp_inputs(lat_active=False, v_ego=0.0, prev=CORNER, curve_memory_enabled=curve_memory))
  for _ in range(stop_frames):                                # hold stopped; pipeline zeroes prev
    p.update(mp_inputs(lat_active=False, v_ego=0.0, prev=0.0, measured=0.0, curve_memory_enabled=curve_memory))


def _launch(p: ModelPathProcessor, curve_memory: bool) -> float:
  # launch: low speed + gated vision (high path std) + conservative ("forgotten") raw curvature
  r = p.update(mp_inputs(lat_active=True, v_ego=3.0, desired=0.005, measured=0.005, prev=0.0,
                         y_std=1.6, curve_memory_enabled=curve_memory))
  return float(r.desired_curvature)


def test_prior_captured_on_transition_only_when_enabled():
  on = ModelPathProcessor()
  on.update(mp_inputs(lat_active=True, curve_memory_enabled=True))
  on.update(mp_inputs(lat_active=False, v_ego=0.0, prev=CORNER, curve_memory_enabled=True))
  assert on._standstill_prior_curvature == CORNER

  off = ModelPathProcessor()
  off.update(mp_inputs(lat_active=True, curve_memory_enabled=False))
  off.update(mp_inputs(lat_active=False, v_ego=0.0, prev=CORNER, curve_memory_enabled=False))
  assert off._standstill_prior_curvature is None        # parity with the old full reset


def test_launch_holds_corner_when_enabled_vs_forgets_when_disabled():
  on = ModelPathProcessor()
  _drive_corner_then_stop(on, curve_memory=True)
  held = _launch(on, curve_memory=True)

  off = ModelPathProcessor()
  _drive_corner_then_stop(off, curve_memory=False)
  forgot = _launch(off, curve_memory=False)

  assert held > 0.012                 # launch curvature stays near the remembered corner (0.02)
  assert forgot < 0.005               # cold start on the conservative raw vision
  assert held > 3.0 * forgot          # curve memory clearly resumes the corner


def test_prior_expires_after_max_standstill():
  p = ModelPathProcessor()
  p.update(mp_inputs(lat_active=True, curve_memory_enabled=True))
  p.update(mp_inputs(lat_active=False, v_ego=0.0, prev=CORNER, curve_memory_enabled=True))
  assert p._standstill_prior_curvature == CORNER
  for _ in range(int(STANDSTILL_PRIOR_MAX_SECONDS / DT) + 5):
    p.update(mp_inputs(lat_active=False, v_ego=0.0, prev=0.0, measured=0.0, curve_memory_enabled=True))
  assert p._standstill_prior_curvature is None
  assert _launch(p, curve_memory=True) < 0.005           # long stop -> forgotten, no resume


def test_prior_cleared_when_rolling_away_inactive():
  # lat inactive but moving (manual drive-away, not a clean stop) -> the corner is stale, forget it
  p = ModelPathProcessor()
  p.update(mp_inputs(lat_active=True, curve_memory_enabled=True))
  p.update(mp_inputs(lat_active=False, v_ego=0.0, prev=CORNER, curve_memory_enabled=True))
  p.update(mp_inputs(lat_active=False, v_ego=6.0, prev=0.0, measured=0.0, curve_memory_enabled=True))
  assert p._standstill_prior_curvature is None


def test_straight_stop_not_remembered():
  p = ModelPathProcessor()
  p.update(mp_inputs(lat_active=True, desired=0.001, measured=0.001, prev=0.001, curve_memory_enabled=True))
  p.update(mp_inputs(lat_active=False, v_ego=0.0, prev=0.001, curve_memory_enabled=True))  # below 0.008
  assert p._standstill_prior_curvature is None
