import pytest

from openpilot.tools.drive_lab.audit_governor_slew import (
  ReversalCandidate,
  _cluster_reversals,
  _event_impulse,
  _sort_log_paths,
)
from openpilot.tools.drive_lab.replay_output_governor import GovernorFrame, ReplaySample, _replay


def _sample(t: float, nominal: float, output: float) -> ReplaySample:
  return ReplaySample(t, True, nominal, output, -output, 0, 1.0, 0, 1.0)


def test_numeric_segment_sort():
  paths = [
    "/tmp/route--10/rlog.zst",
    "/tmp/route--2/rlog.zst",
    "/tmp/route--1/rlog.zst",
  ]
  assert _sort_log_paths(paths) == [paths[2], paths[1], paths[0]]


def test_reversal_clustering_keeps_non_overlapping_strongest_events():
  candidates = [
    ReversalCandidate(10, 1.00, 1, 0.10),
    ReversalCandidate(20, 1.20, -1, 0.30),
    ReversalCandidate(30, 2.00, 1, 0.20),
  ]
  selected = _cluster_reversals(candidates, spacing_s=0.75)
  assert [candidate.index for candidate in selected] == [20, 30]
  assert all(selected[index].t - selected[index - 1].t >= 0.75 for index in range(1, len(selected)))


def test_event_impulse_is_positive_and_uses_fixed_dt():
  g0 = [_sample(i * 0.01, 0.2, output) for i, output in enumerate((0.3, 0.2, 0.1))]
  g2 = [_sample(i * 0.01, 0.2, output) for i, output in enumerate((0.1, 0.1, 0.1))]
  assert _event_impulse(g0, g2, 0, horizon_ticks=3) == pytest.approx(0.003)


def test_same_direction_negative_control_has_no_false_g1_difference():
  frames = []
  for i in range(20):
    nominal = 0.2 if (i // 3) % 2 == 0 else -0.2
    frames.append(GovernorFrame(
      t=i * 0.01, active=True, nominal_torque=nominal, logged_output=-nominal,
      logged_reason=0, logged_cap=1.0, v_ego=20.0, steering_rate_deg=0.0,
      steering_pressed=False, desired_lateral_accel=0.0, actual_lateral_accel=0.0,
      same_direction_limit=False, controller_evidence_stable=True, path_evidence_valid=True,
      lateral_accel_error_rate=0.0, lat_delay=0.01,
    ))
  replay = _replay(frames)
  assert all(g0.output_torque == pytest.approx(g1.output_torque)
             for g0, g1 in zip(replay["G0"], replay["G1"], strict=True))
