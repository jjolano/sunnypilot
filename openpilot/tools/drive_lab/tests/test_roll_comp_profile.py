import json
import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.custom.lateral.block_jackknife import (
  fit_block_slope,
)
from openpilot.tools.drive_lab import roll_comp_profile as profile_module
from openpilot.tools.drive_lab.roll_comp_profile import (
  GRAVITY,
  ROLL_COMP_SPEED_BANDS,
  ROLL_COMP_VERDICT_MIN_ROLL_SPAN,
  RollCompProfileReport,
  _Frame,
  _collect_evidence,
  _quality_reason,
  _select_straight_frames,
  build_roll_comp_band_reports,
  build_roll_comp_profile,
  build_roll_comp_verdict_report,
  load_roll_comp_profile,
  render_roll_comp_profile,
  render_roll_comp_verdict_report,
  save_roll_comp_profile,
)


class FakeMsg(SimpleNamespace):
  def which(self):
    return self.kind


class FakeUnion(SimpleNamespace):
  def which(self):
    return 'torqueState'


def _msg(kind: str, t_s: float, **payload):
  return FakeMsg(kind=kind, logMonoTime=int(t_s * 1e9), **{kind: SimpleNamespace(**payload)})


def _frame(t_s: float, *, x: float = 0.0, slope: float = 0.55, integrator: float = 0.0,
           desired: float = 0.0, v_ego: float = 20.0, lat_active: bool = True,
           steering_pressed: bool = False, saturated: bool = False, torque_active: bool = True):
  return _Frame(
    t_s=t_s,
    lat_active=lat_active,
    steering_pressed=steering_pressed,
    v_ego=v_ego,
    roll=-math.asin(x / GRAVITY),
    p=slope * x,
    i=integrator,
    f=0.0,
    desired_lateral_accel=desired,
    saturated=saturated,
    torque_active=torque_active,
  )


def _block_frames(block_id, *, slope=0.55, xs=None, count=240, desired=0.0, v_ego=20.0):
  if xs is None:
    xs = np.linspace(-0.4, 0.4, count)
  return [
    _frame(block_id * 65.0 + index * 0.25, x=float(xs[index % len(xs)]), slope=slope,
           integrator=0.1 + block_id * 0.01, desired=desired, v_ego=v_ego)
    for index in range(count)
  ]


def _completed_frames(blocks=12, *, slope=0.55, xs=None, count=240, desired=0.0, v_ego=20.0):
  frames = []
  for block_id in range(blocks):
    frames.extend(_block_frames(block_id, slope=slope, xs=xs, count=count, desired=desired, v_ego=v_ego))
  frames.append(_frame(60.0 + (blocks - 1) * 65.0, x=0.0, slope=slope))
  return frames


def _messages(frames):
  messages = []
  for frame in frames:
    torque_state = SimpleNamespace(
      active=frame.torque_active,
      p=frame.p,
      i=frame.i,
      f=frame.f,
      desiredLateralAccel=frame.desired_lateral_accel,
      saturated=frame.saturated,
    )
    messages.extend([
      _msg('carState', frame.t_s, vEgo=frame.v_ego, steeringPressed=frame.steering_pressed),
      _msg('carControl', frame.t_s, latActive=frame.lat_active),
      _msg('liveParameters', frame.t_s, roll=frame.roll),
      _msg('controlsState', frame.t_s, lateralControlState=FakeUnion(torqueState=torque_state)),
    ])
  return messages


def _report_from_frames(frames, *, strict_straight=False, v_lo=15.0, v_hi=None):
  rows, clock = _collect_evidence(frames, strict_straight, v_lo, v_hi)
  completed_rows = [row for row in rows if clock.is_completed(row[2])]
  points = np.asarray([row[:3] for row in completed_rows], dtype=float)
  if not len(points):
    points = np.empty((0, 3), dtype=float)
  informative = profile_module._informative_blocks(points) if len(points) else []
  informative_points = points[np.isin(points[:, 2].astype(int), informative)] if informative else np.empty((0, 3))
  fit = fit_block_slope(informative_points) if len(informative_points) else None
  roll_span = float(np.ptp(informative_points[:, 0])) if len(informative_points) else 0.0
  valid, reason = _quality_reason(informative_points, informative, fit, roll_span)
  return RollCompProfileReport(
    source='analytic',
    slope=float(fit.slope) if fit is not None and math.isfinite(fit.slope) else None,
    integrator_mean=float(np.mean([row[3] for row in completed_rows])) if completed_rows else None,
    integrator_std=float(np.std([row[3] for row in completed_rows])) if completed_rows else None,
    point_count=len(informative_points),
    roll_span=roll_span,
    slope_rel_se=float(fit.rel_se) if fit is not None and math.isfinite(fit.rel_se) else None,
    block_count=len(informative),
    quality_valid=valid,
    quality_reason=reason,
  )


def _quality_report(source, slope, *, spread=0.4, rel_se=0.05, blocks=12, points=2400, valid=True):
  return RollCompProfileReport(
    source=source,
    slope=slope,
    integrator_mean=0.0,
    integrator_std=0.01,
    point_count=points,
    roll_span=spread,
    slope_rel_se=rel_se,
    block_count=blocks,
    quality_valid=valid,
    quality_reason='all temporal-block quality gates passed' if valid else 'synthetic failure',
  )


def test_public_build_recovers_clean_completed_block_profile():
  frames = _completed_frames()
  report = build_roll_comp_profile(_messages(frames), source='synthetic', already_sorted=True)

  assert report.quality_valid
  assert report.quality_reason == 'all temporal-block quality gates passed'
  assert report.slope == pytest.approx(0.55, abs=1e-6)
  assert report.slope_rel_se == pytest.approx(0.0)
  assert report.block_count == 12
  assert report.point_count == 12 * 240
  assert report.roll_span > ROLL_COMP_VERDICT_MIN_ROLL_SPAN
  assert report.integrator_mean is not None
  assert 'slope rel SE' in render_roll_comp_profile(report)


def test_timestamp_spaced_opportunities_skip_missed_slots_and_guards():
  frames = [
    _frame(0.0, x=-0.2),
    _frame(0.8, x=0.2),
    _frame(0.9, x=0.3),
    _frame(1.6, x=0.4),
  ]
  rows, clock = _collect_evidence(frames, False, 15.0, None)
  assert [row[0] for row in rows] == pytest.approx([-0.2, 0.2, 0.4])

  guard_frames = _completed_frames(blocks=1) + [_frame(60.25, x=0.3), _frame(60.5, x=-0.3)]
  guard_rows, guard_clock = _collect_evidence(guard_frames, False, 15.0, None)
  assert all(row[2] == 0 for row in guard_rows)
  assert guard_clock.completed_through == 0

  incomplete = _block_frames(0) + _block_frames(1, count=20)
  incomplete_rows, incomplete_clock = _collect_evidence(incomplete, False, 15.0, None)
  assert {row[2] for row in incomplete_rows} == {0, 1}
  assert {row[2] for row in incomplete_rows if incomplete_clock.is_completed(row[2])} == {0}
  assert incomplete_clock.completed_through == 0


def test_original_desired_delta_semantics_update_on_non_opportunity_frame():
  frames = [
    _frame(0.0, x=0.0, desired=0.0),
    _frame(0.05, x=0.1, desired=0.10),
    _frame(0.10, x=0.2, desired=0.10),
    _frame(0.25, x=0.3, desired=0.10),
  ]
  rows, _clock = _collect_evidence(frames, False, 15.0, None)

  # The last opportunity settled after the change on a non-opportunity frame;
  # it must compare against that settled value, not only against t=0.
  assert len(rows) == 2


def test_select_straight_frames_preserves_original_safety_gates():
  frames = [
    _frame(0.0, x=0.1, lat_active=False),
    _frame(0.05, x=0.1, steering_pressed=True),
    _frame(0.10, x=0.1, v_ego=10.0),
    _frame(0.15, x=0.1, desired=0.5),
    _frame(0.20, x=0.1, saturated=True),
    _frame(0.25, x=0.1),
  ]
  selected = _select_straight_frames(frames)
  assert len(selected) == 1


def test_select_straight_frames_delta_gate_excludes_large_transition():
  frames = [
    _frame(0.0, x=0.0, desired=0.0),
    _frame(0.05, x=0.1, desired=0.10),
    _frame(0.10, x=0.2, desired=0.10),
  ]
  selected = _select_straight_frames(frames)
  assert [frame.t_s for frame in selected] == [0.0, 0.10]


def test_strict_straight_tightens_desired_accel_gate():
  loose = [_frame(i * 0.05, x=0.1, desired=0.10) for i in range(12)]
  strict_allowed = [_frame(i * 0.05, x=0.1, desired=0.05) for i in range(12)]
  assert len(_select_straight_frames(loose)) == 12
  assert len(_select_straight_frames(loose, strict_straight=True)) == 0
  assert len(_select_straight_frames(strict_allowed, strict_straight=True)) == 12


def test_quality_rejects_correlated_noise_and_failure_modes():
  noisy_xs = np.array([-0.45, -0.30, -0.10, 0.10, 0.30, 0.45])
  narrow_xs = np.concatenate((np.linspace(-0.45, 0.45, 40), np.array([-0.06, 0.06] * 100)))
  noisy_frames = _block_frames(0, slope=0.3, xs=noisy_xs) + [
    frame for block_id in range(1, 12) for frame in _block_frames(block_id, slope=1.0, xs=narrow_xs)
  ]
  noisy_frames.append(_frame(775.0, x=0.0))
  noisy = _report_from_frames(noisy_frames)
  assert noisy.slope is not None
  assert noisy.slope_rel_se is not None and noisy.slope_rel_se > 1 / 3
  assert not noisy.quality_valid
  assert 'relative SE' in noisy.quality_reason

  cases = [
    ('degenerate', _completed_frames(xs=np.zeros(240)), 'informative completed block'),
    ('negative', _completed_frames(slope=-0.55), 'non-positive'),
    ('full-out-of-range', _completed_frames(slope=1.2), 'outside'),
  ]
  for _name, frames, reason in cases:
    report = _report_from_frames(frames)
    assert not report.quality_valid
    assert reason in report.quality_reason


def test_quality_rejects_leave_one_out_out_of_range_and_insufficient_blocks():
  frames = _block_frames(0, slope=0.55, xs=np.array([-0.45, -0.30, -0.10, 0.10, 0.30, 0.45]))
  spread_xs = np.concatenate((np.linspace(-0.45, 0.45, 40), np.array([-0.06, 0.06] * 100)))
  frames += [
    frame for block_id in range(1, 12)
    for frame in _block_frames(block_id, slope=1.2, xs=spread_xs)
  ]
  frames.append(_frame(775.0, x=0.0))
  report = _report_from_frames(frames)
  assert not report.quality_valid
  assert 'leave-one-block-out slope' in report.quality_reason

  eleven = _completed_frames(blocks=11)
  insufficient = _report_from_frames(eleven)
  assert insufficient.block_count == 11
  assert not insufficient.quality_valid
  assert 'only 11 informative' in insufficient.quality_reason

  incomplete = _report_from_frames(_block_frames(0) + _block_frames(1, count=20))
  assert incomplete.block_count == 1
  assert not incomplete.quality_valid
  assert 'only 1 informative' in incomplete.quality_reason


def test_discarded_blocks_do_not_inflate_route_roll_span_gate():
  frames = []
  narrow = np.linspace(-0.14, 0.14, 240)
  for block_id in range(12):
    frames.extend(_block_frames(block_id, xs=narrow))
  frames.extend(_block_frames(12, xs=np.linspace(-0.5, 0.5, 10), count=10))
  frames.append(_frame(840.0, x=0.0))
  report = _report_from_frames(frames)

  assert report.block_count == 12
  assert report.point_count == 12 * 240
  # Only the discarded block pushes the all-point span beyond 0.3.
  assert report.roll_span < 0.3
  assert not report.quality_valid


def test_report_summaries_and_quality_fields_are_consistent():
  report = _report_from_frames(_completed_frames())
  assert report.point_count == report.block_count * 240
  assert report.slope_rel_se is not None and report.slope_rel_se <= 1 / 3
  assert report.quality_valid
  payload = report.to_dict()
  assert payload['slope_rel_se'] == report.slope_rel_se
  assert payload['block_count'] == report.block_count
  assert payload['quality_valid'] is True


def test_report_save_load_render_round_trip_and_legacy_fail_closed(tmp_path):
  report = _report_from_frames(_completed_frames())
  path = tmp_path / 'roll-comp-profile.json'
  save_roll_comp_profile(report, path)
  loaded = profile_module.load_roll_comp_profile(path)
  assert loaded == report
  rendered = render_roll_comp_profile(loaded)
  assert 'block count:' in rendered
  assert 'quality valid:' in rendered
  assert 'quality reason:' in rendered

  optional = report.to_dict()
  optional.update({
    'slope': None,
    'integrator_mean': None,
    'integrator_std': None,
    'slope_rel_se': None,
    'quality_valid': False,
    'quality_reason': 'not enough evidence',
  })
  optional_path = tmp_path / 'optional.json'
  optional_path.write_text(json.dumps(optional))
  optional_loaded = load_roll_comp_profile(optional_path)
  assert optional_loaded.slope is None
  assert optional_loaded.integrator_mean is None
  assert optional_loaded.integrator_std is None
  assert optional_loaded.slope_rel_se is None
  assert not optional_loaded.quality_valid

  legacy_path = tmp_path / 'legacy.json'
  legacy_path.write_text(json.dumps({
    'source': 'legacy', 'slope': 0.55, 'integrator_mean': 0.0,
    'integrator_std': 0.01, 'point_count': 3000, 'roll_span': 0.5,
  }))
  legacy = load_roll_comp_profile(legacy_path)
  assert legacy.block_count == 0
  assert legacy.slope_rel_se is None
  assert not legacy.quality_valid
  assert build_roll_comp_verdict_report([legacy, legacy, legacy]).verdict == 'insufficient-data'


def test_saved_report_parser_is_strict_and_fail_closed(tmp_path):
  valid = _quality_report('valid', 0.55).to_dict()

  malformed_payloads = []
  for field in ('slope', 'integrator_mean', 'integrator_std', 'roll_span', 'slope_rel_se', 'point_count', 'block_count'):
    payload = dict(valid)
    payload[field] = str(payload[field])
    malformed_payloads.append(payload)
  for field in ('point_count', 'block_count'):
    for value in (True, 1.5, -1):
      payload = dict(valid)
      payload[field] = value
      malformed_payloads.append(payload)
  for field in ('slope', 'integrator_mean', 'integrator_std', 'roll_span', 'slope_rel_se', 'point_count', 'block_count'):
    for value in (float('nan'), float('inf'), float('-inf')):
      payload = dict(valid)
      payload[field] = value
      malformed_payloads.append(payload)
  for field in ('slope_rel_se', 'block_count', 'quality_valid', 'quality_reason'):
    payload = dict(valid)
    del payload[field]
    malformed_payloads.append(payload)

  for value in ('false', 1):
    payload = dict(valid)
    payload['quality_valid'] = value
    malformed_payloads.append(payload)
  for value in ('', None, 1):
    payload = dict(valid)
    payload['quality_reason'] = value
    malformed_payloads.append(payload)
  for value in (1, None):
    payload = dict(valid)
    payload['source'] = value
    malformed_payloads.append(payload)

  paths = []
  for index, payload in enumerate(malformed_payloads):
    path = tmp_path / f'malformed-{index}.json'
    path.write_text(json.dumps(payload, allow_nan=True))
    paths.append(path)
  root = tmp_path / 'root.json'
  root.write_text('[]')
  paths.append(root)
  broken = tmp_path / 'broken.json'
  broken.write_text('{')
  paths.append(broken)
  empty = tmp_path / 'empty.json'
  empty.write_text('')
  paths.append(empty)

  for path in paths:
    report = load_roll_comp_profile(path)
    assert not report.quality_valid
    assert report.quality_reason.startswith('malformed saved report:')
    assert build_roll_comp_verdict_report([report, report, report]).verdict == 'insufficient-data'


def test_saved_report_empty_source_is_malformed(tmp_path):
  payload = _quality_report('', 0.55).to_dict()
  path = tmp_path / 'empty-source.json'
  path.write_text(json.dumps(payload))

  report = load_roll_comp_profile(path)

  assert not report.quality_valid
  assert report.quality_reason.startswith('malformed saved report:')


def test_route_verdict_requires_distinct_nonempty_normalized_sources():
  same = _quality_report('same', 0.50)
  assert build_roll_comp_verdict_report([same] * 3).qualifying_route_count == 1
  assert build_roll_comp_verdict_report([same, replace(same), replace(same)]).verdict == 'insufficient-data'

  whitespace = [replace(same, source=' route '), replace(same, source='route'), replace(same, source='  route')]
  assert build_roll_comp_verdict_report(whitespace).qualifying_route_count == 1
  assert build_roll_comp_verdict_report([replace(same, source=' '), replace(same, source='  '), replace(same, source='\t')]).qualifying_route_count == 0
  assert build_roll_comp_verdict_report([replace(same, source=''), replace(same, source='  '), same]).qualifying_route_count == 1

  routes = [_quality_report(' route-a ', 0.50), _quality_report('route-b', 0.52), _quality_report('route-c', 0.53)]
  promoted = build_roll_comp_verdict_report(routes)
  assert promoted.qualifying_route_count == 3
  assert promoted.slope_spread == pytest.approx(0.03)
  assert promoted.verdict == 'promote'

  duplicate_source = [
    _quality_report('route-a', 0.50), _quality_report(' route-a ', 0.90),
    _quality_report('route-b', 0.52), _quality_report('route-c', 0.53),
  ]
  deduped = build_roll_comp_verdict_report(duplicate_source)
  assert deduped.qualifying_route_count == 3
  assert deduped.slope_spread == pytest.approx(0.03)


@pytest.mark.parametrize('v_lo,v_hi', ROLL_COMP_SPEED_BANDS)
def test_offline_speed_bounds_match_live_inclusive_lower_exclusive_upper(v_lo, v_hi):
  lower = _frame(0.0, x=0.1, v_ego=v_lo)
  upper = _frame(0.0, x=0.1, v_ego=v_hi)
  from openpilot.tools.drive_lab.roll_comp_profile import _passes_straight_gates

  assert _passes_straight_gates(lower, True, False, v_lo, v_hi)
  assert not _passes_straight_gates(upper, True, False, v_lo, v_hi)

  rows, _clock = _collect_evidence([lower, upper], False, v_lo, v_hi)
  assert len(rows) == 1


def test_malformed_quality_values_cannot_promote():
  valid = _quality_report('valid', 0.55)
  malformed_rel_se = replace(valid, slope_rel_se=float('nan'))
  malformed_blocks = replace(valid, block_count=0)
  assert build_roll_comp_verdict_report([malformed_rel_se] * 3).verdict == 'insufficient-data'
  assert build_roll_comp_verdict_report([malformed_blocks] * 3).verdict == 'insufficient-data'


def test_verdict_requires_three_quality_routes_and_low_gain_spread():
  routes = [_quality_report('a', 0.50, spread=0.42), _quality_report('b', 0.52, spread=0.41),
            _quality_report('c', 0.53, spread=0.40)]
  report = build_roll_comp_verdict_report(routes)
  assert report.routes == routes
  assert report.qualifying_route_count == 3
  assert report.slope_spread == pytest.approx(0.03)
  assert report.verdict == 'promote'
  assert 'verdict: promote' in render_roll_comp_verdict_report(report)

  assert build_roll_comp_verdict_report(routes[:2]).verdict == 'insufficient-data'
  high_se = replace(routes[0], slope_rel_se=0.34)
  assert build_roll_comp_verdict_report([high_se, routes[1], routes[2]]).verdict == 'insufficient-data'
  parked = build_roll_comp_verdict_report([
    _quality_report('a', 0.50), _quality_report('b', 0.55), _quality_report('c', 0.60),
  ])
  assert parked.slope_spread == pytest.approx(0.10)
  assert parked.verdict == 'park'
  assert build_roll_comp_verdict_report([
    _quality_report('a', 0.50), _quality_report('b', 0.52), _quality_report('c', 0.55),
  ]).verdict == 'park'


def test_speed_band_reports_share_quality_path_and_alignment():
  frames = []
  for block_id in range(12):
    xs = np.linspace(-0.4, 0.4, 240)
    for index, x in enumerate(xs):
      speed = (7.0, 12.0, 20.0)[index % 3]
      frames.append(_frame(block_id * 65.0 + index * 0.25, x=float(x), v_ego=speed))
  frames.append(_frame(775.0, x=0.0))
  reports = build_roll_comp_band_reports(_messages(frames), source='bands', already_sorted=True)

  assert [report.source for report in reports] == [
    'bands [5-10 m/s]', 'bands [10-15 m/s]', 'bands [15-100 m/s]',
  ]
  assert [report.block_count for report in reports] == [12, 12, 12]
  assert [report.point_count for report in reports] == [960, 960, 960]
  assert all(not report.quality_valid for report in reports)


def test_cli_speed_bands_public_mode(monkeypatch, capsys):
  frames = _completed_frames(blocks=1)
  messages = _messages(frames)
  monkeypatch.setattr(profile_module, 'load_route_msgs', lambda *_args, **_kwargs: messages)
  monkeypatch.setattr(sys, 'argv', ['roll_comp_profile', 'route', '--speed-bands', '--json'])

  profile_module.main()
  payload = json.loads(capsys.readouterr().out)

  assert len(payload) == 3
  assert all('slope_rel_se' in report and 'block_count' in report for report in payload)


def test_cli_single_route_json_mode(monkeypatch, capsys):
  messages = _messages(_completed_frames(blocks=1))
  monkeypatch.setattr(profile_module, 'load_route_msgs', lambda *_args, **_kwargs: messages)
  monkeypatch.setattr(sys, 'argv', ['roll_comp_profile', 'route', '--json'])

  profile_module.main()
  payload = json.loads(capsys.readouterr().out)

  assert payload['source'] == 'route'
  assert 'quality_valid' in payload
  assert 'quality_reason' in payload
